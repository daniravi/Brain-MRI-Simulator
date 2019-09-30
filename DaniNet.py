from __future__ import division
import os
import time
from glob import glob
from ops import *
import tensorflow as tf
import h5py
import pickle
import skfuzzy
import random


class DaniNet(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=128,  # size the input images
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=100,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=1,  # number of channels of input images
                 num_encoder_channels=64,  # number of channels of the first conv layer of encoder
                 num_z_channels=200,  # number of channels of the layer z (noise or code)
                 num_progression_points=10,  # number of progression classes
                 num_gen_channels=1024,  # number of channels of the first deconv layer of generator
                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 save_dir='./save',  # path to save checkpoints, samples, and summary
                 output_dir='sample_TR',  # name for the folder where sample are generated
                 dataset_name='TrainingSetMRI',  # name of the dataset in the folder ./data
                 slice_number=100,  # current MRI slice to consider
                 max_regional_expansion=10,
                 map_disease=(0, 1, 2, 2, 2, 3),
                 age_intervals=(63, 66, 68, 70, 72, 74, 76, 78, 80, 83, 87)
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.input_value_range = (0, 255)
        self.ration_input_output_range = (self.input_value_range[1] - self.input_value_range[0]) / (self.image_value_range[1] - self.image_value_range[0])
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_progression_points = num_progression_points
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.slice_number = slice_number
        self.age_intervals = age_intervals
        self.map_disease = map_disease
        self.max_regional_expansion = max_regional_expansion  # max rate of regional expansion
        self.save_dir = save_dir + '/' + str(slice_number)
        self.output_dir = output_dir
        self.dataset_name = dataset_name + '/' + str(slice_number)
        self.regressors = []
        self.rescales = []
        self.allMask = []
        self.summary = []
        self.EG_learning_rate_summary = []
        self.D_z_optimizer = []
        self.EG_optimizer = []
        self.D_img_optimizer = []
        self.writer = []
        self.loss_EG = 0
        self.loss_Dz = 0
        self.loss_Di = 0
        self.D_img_loss_G = 0
        self.G_img_loss = 0
        self.n_of_regions = 0
        self.numb_of_sample = int(np.sqrt(size_batch))
        self.n_of_diagnosis = np.size(np.unique(map_disease))

        # ****************************************** Framework Parameters ***************************************************
        self.bin_variance_scale = 0.2 #this is connected with the first parameter of loss_weight
        self.minimum_input_similarity = 0.3 #this is connected with the first parameter of loss_weight
        self.n_regions_can_be_processed = 1  # max number of random region that can be processed at each iteration
        self.loss_weight = (0.5, 0.5, 0.5, 0.15, 0.15)
        # 0)similarity with the image # the sum need to be equal to 1 (population avarage vs personality)
        # 1)realistic image (smaller is more realistic structures)   (realistic structure vs number of epoch)
        # 2)smoothing in progression (0 very smooth , 1 major freedom to be different), (temporal smooting vs progression)
        # 3)pixel loss  (progression)
        # 4)regional loss (progression-> reliability of progression prior)

        self.loss_weight_scale= (10**2, 10**-5, 10**-2, 10, 10)
        self.minimum_input_similarity_scale=10**-13
        self.loss_weight = np.multiply(self.loss_weight,self.loss_weight_scale)
        self.minimum_input_similarity=self.minimum_input_similarity*self.minimum_input_similarity_scale

        # *********************************************************************************************
        self.bin_centers = np.convolve(self.age_intervals, [0.5, 0.5], 'valid')
        self.bin_size = np.diff(self.bin_centers)
        self.bin_size = np.append(self.bin_size, self.bin_size[self.bin_size.size - 1])
        self.bin_centers_tensor = tf.constant(self.bin_centers)

        # ************************************* input to graph ********************************************************
        self.input_image = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_images'
        )

        self.age = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_progression_points],
            name='age_labels'
        )

        self.age_index = tf.placeholder(
            tf.int32,
            [self.size_batch],
            name='age_index_labels'
        )
        self.fuzzy_membership = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_progression_points],
            name='fuzzy_membership_label'
        )
        self.diagnosis = tf.placeholder(
            tf.float32,
            [self.size_batch, self.n_of_diagnosis],
            name='diagnosis_labels'
        )

        self.z_prior = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_z_channels],
            name='z_prior'
        )
        # ************************************* build the graph *******************************************************
        print('\n\tBuilding graph ...')

        # encoder: input image --> z
        self.z = self.encoder(
            image=self.input_image
        )

        # generator: z + label --> generated image
        self.G = self.generator(
            z=self.z,
            y=self.age,
            diagnosis=self.diagnosis,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio
        )

        # discriminator on z
        self.D_z, self.D_z_logits = self.discriminator_z(
            z=self.z,
            is_training=self.is_training
        )

        # discriminator on G
        self.D_G, self.D_G_logits = self.discriminator_img(
            image=self.G,
            y=self.age,
            diagnosis=self.diagnosis,
            is_training=self.is_training
        )

        # discriminator on z_prior
        self.D_z_prior, self.D_z_prior_logits = self.discriminator_z(
            z=self.z_prior,
            is_training=self.is_training,
            reuse_variables=True
        )

        # discriminator on input image
        self.D_input, self.D_input_logits = self.discriminator_img(
            image=self.input_image,
            y=self.age,
            diagnosis=self.diagnosis,
            is_training=self.is_training,
            reuse_variables=True
        )

        self.load_extra_data()

        self.pixel_regres_loss, self.region_regres_loss, self.deformation_loss = self.longitudinal_constrains(
            images=self.input_image,
            diagnosis=self.diagnosis,
            age_index=self.age_index,
            fuzzy_membership=self.fuzzy_membership
        )

        # loss function of discriminator on z
        self.D_z_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_prior_logits,
                                                    labels=tf.ones_like(self.D_z_prior_logits))
        )
        self.D_z_loss_z = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_logits, labels=tf.zeros_like(self.D_z_logits))
        )
        self.E_z_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_logits, labels=tf.ones_like(self.D_z_logits))
        )
        # loss function of discriminator on image
        self.D_img_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_input_logits,
                                                    labels=tf.ones_like(self.D_input_logits))
        )

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        # variables of encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        # variables of generator
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        # variables of discriminator on z
        self.D_z_variables = [var for var in trainable_variables if 'D_z_' in var.name]
        # variables of discriminator on image
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]

        # ************************************* collect the summary ***************************************
        self.z_summary = tf.summary.histogram('z', self.z)
        self.z_prior_summary = tf.summary.histogram('z_prior', self.z_prior)
        self.pixel_regres_summary = tf.summary.scalar('pixel_regres_loss', self.pixel_regres_loss)
        self.region_regres_summary = tf.summary.scalar('region_regres_loss', self.region_regres_loss)
        self.deformation_summary = tf.summary.scalar('deformation_loss', self.deformation_loss)
        self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
        self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
        self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
        self.E_z_loss_summary = tf.summary.scalar('E_z_loss', self.E_z_loss)
        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self,
              num_epochs=200,  # number of epochs
              learning_rate=0.0003,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # use the saved checkpoint to initialize the network
              use_init_model=True,  # use the init model to initialize the network
              n_epoch_to_save=3,
              conditioned_enabled=True,
              progression_enabled=True
              ):
        weights = self.loss_weight
        file_names = glob(os.path.join('./data', self.dataset_name, '*.png'))
        size_data = len(file_names)
        np.random.seed(seed=2017)
        if enable_shuffle:
            np.random.shuffle(file_names)
        if progression_enabled:
            self.loss_EG = self.G_img_loss * weights[1] + self.E_z_loss * weights[2] + \
                           self.pixel_regres_loss * weights[3] + self.region_regres_loss * weights[4] + \
                           self.deformation_loss * weights[0]
        else:
            self.loss_EG = self.G_img_loss * weights[1] + self.E_z_loss * weights[2] + \
                           self.deformation_loss * weights[0]

        self.loss_Dz = self.D_z_loss_prior + self.D_z_loss_z
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G

        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=size_data / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        # optimizer for encoder + generator
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            self.EG_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_EG,
                global_step=self.EG_global_step,
                var_list=self.E_variables + self.G_variables
            )

            # optimizer for discriminator on z
            self.D_z_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Dz,
                var_list=self.D_z_variables
            )

            # optimizer for discriminator on image
            self.D_img_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Di,
                var_list=self.D_img_variables
            )

        # *********************************** tensorboard *************************************************************
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([self.pixel_regres_summary, self.region_regres_summary, self.deformation_summary,
                                         self.D_img_loss_input_summary, self.D_img_loss_G_summary,
                                         self.G_img_loss_summary, self.EG_learning_rate_summary,
                                         self.D_G_logits_summary, self.D_input_logits_summary, self.E_z_loss_summary
                                         ])

        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)


        # ******************************************* training *******************************************************
        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")
                # load init model
                if use_init_model:
                    if not os.path.exists('init_model/model-init.data-00000-of-00001'):
                        from init_model.zip_opt import join
                        try:
                            join('init_model/model_parts', 'init_model/model-init.data-00000-of-00001')
                        except:
                            raise Exception('Error joining files')
                    self.load_checkpoint(model_path='init_model')

        num_batches = len(file_names) // self.size_batch

        batch_images = []
        batch_label_age = []
        batch_fuzzy_membership = []
        batch_label_age_index = []
        batch_label_diagnosis = []
        batch_z_prior = []
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(file_names)
            start_time = time.time()
            for ind_batch in range(num_batches):
                # read batch images and labels
                batch_files = file_names[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch = [load_image(
                    image_path=batch_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for batch_file in batch_files]
                if self.num_input_channels == 1:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_label_age = np.ones(
                    shape=(len(batch_files), self.num_progression_points),
                    dtype=np.float
                ) * self.image_value_range[0]
                batch_fuzzy_membership = np.ones(
                    shape=(len(batch_files), self.num_progression_points),
                    dtype=np.float
                ) * self.image_value_range[0]
                batch_label_age_index = np.ones(
                    shape=(len(batch_files)),
                    dtype=np.int
                ) * self.image_value_range[0]

                batch_label_diagnosis = np.ones(
                    shape=(len(batch_files), self.n_of_diagnosis),
                    dtype=np.float
                ) * self.image_value_range[0]
                for i, label in enumerate(batch_files):
                    real_age = float(str(batch_files[i]).split('/')[-1].split('_')[0])
                    age_index = np.min([np.max([np.digitize(real_age, self.age_intervals) - 1, 0]), 9])

                    batch_label_age_index[i] = age_index
                    batch_label_age[i, age_index] = self.image_value_range[-1]

                    # curr_gender = int(str(batch_files[i]).split('/')[-1].split('_')[1])
                    for t in range(self.num_progression_points):
                        if progression_enabled:
                            batch_fuzzy_membership[i, t] = skfuzzy.membership.gaussmf(real_age, self.bin_centers[t],
                                                                                      np.sqrt(self.bin_size[t]) * self.bin_variance_scale)
                        else:
                            batch_fuzzy_membership[i, label] = self.image_value_range[-1]
                    if not use_init_model:
                        curr_diagnosis = random.randint(0, max(self.map_disease))
                    else:
                        curr_diagnosis = self.map_disease[int(str(batch_files[i]).split('/')[-1].split('_')[2]) - 1]
                    if conditioned_enabled:
                        batch_label_diagnosis[i, curr_diagnosis] = self.image_value_range[-1]
                    else:
                        batch_label_diagnosis[i, 0] = self.image_value_range[-1]

                # prior distribution on the prior of z
                batch_z_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_z_channels]
                ).astype(np.float32)

                _, _, _, = self.session.run(
                    fetches=[
                        self.EG_optimizer,
                        self.D_z_optimizer,
                        self.D_img_optimizer
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.age: batch_label_age,
                        self.age_index: batch_label_age_index,
                        self.fuzzy_membership: batch_fuzzy_membership,
                        self.diagnosis: batch_label_diagnosis,
                        self.z_prior: batch_z_prior
                    }
                )
            # add to summary
            summary = self.summary.eval(
                feed_dict={
                    self.input_image: batch_images,
                    self.age: batch_label_age,
                    self.age_index: batch_label_age_index,
                    self.fuzzy_membership: batch_fuzzy_membership,
                    self.diagnosis: batch_label_diagnosis,
                    self.z_prior: batch_z_prior
                }
            )
            self.writer.add_summary(summary, self.EG_global_step.eval())
            print("\nEpoch: [%3d/%3d]\n" % (epoch + 1, num_epochs))
            elapse = time.time() - start_time
            print(time.strftime("Time: %H:%M:%S", time.gmtime(elapse)))
            # save sample images for each epoch
            name = '{:02d}.png'.format(epoch + 1)
            self.test(batch_images, batch_label_diagnosis, name)
            # save checkpoint for each 3 epoch
            if np.mod(epoch, n_epoch_to_save) == n_epoch_to_save - 1:
                self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()

    def load_extra_data(self):
        mat = h5py.File('Mask/Mask_' + str(self.slice_number) + '.mat', 'r')
        self.allMask = mat['allMask'][()]  # array
        self.allMask = self.allMask.T
        self.allMask = tf.convert_to_tensor(self.allMask, np.float32)
        file_handler = open('Regressor/' + str(self.slice_number), 'rb')
        [self.regressors, self.rescales] = pickle.load(file_handler)

        self.n_of_regions = np.size(self.rescales)
        if np.size(np.shape(self.regressors)) == 2:
            tmp = []
            for i in range(self.n_of_regions):
                tmp.append(self.regressors[i][1])
            self.regressors = tmp

    def encoder(self, image, reuse_variables=False):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = conv2d(
                input_map=current,
                num_output_channels=self.num_encoder_channels * (2 ** i),
                size_kernel=self.size_kernel,
                name=name
            )
            current = tf.nn.relu(current)

        # fully connection layer
        name = 'E_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=self.num_z_channels,
            name=name
        )

        # output
        return tf.nn.tanh(current)

    def generator(self, z, y, diagnosis, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / self.num_progression_points)
        else:
            duplicate = 1
        z = concat_label(z, y, duplicate=duplicate)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / 2)
        else:
            duplicate = 1
        z = concat_label(z, diagnosis, duplicate=duplicate)
        size_mini_map = int(self.size_image / 2 ** num_layers)
        # fc layer
        name = 'G_fc'
        current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
            name=name
        )
        # reshape to cube for deconv
        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        current = tf.nn.relu(current)
        # deconv layers with stride 2
        i = 0
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = deconv2d(
                input_map=current,
                output_shape=[self.size_batch,
                              size_mini_map * 2 ** (i + 1),
                              size_mini_map * 2 ** (i + 1),
                              int(self.num_gen_channels / 2 ** (i + 1))],
                size_kernel=self.size_kernel,
                name=name
            )
            current = tf.nn.relu(current)
        name = 'G_deconv' + str(i + 1)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          int(self.num_gen_channels / 2 ** (i + 2))],
            size_kernel=self.size_kernel,
            stride=1,
            name=name
        )
        current = tf.nn.relu(current)
        name = 'G_deconv' + str(i + 2)
        current = deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          self.num_input_channels],
            size_kernel=self.size_kernel,
            stride=1,
            name=name
        )
        return tf.nn.tanh(current)

    @staticmethod
    def discriminator_z(z, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16),
                        enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        current = z
        # fully connection layer
        i = 0
        for i in range(len(num_hidden_layer_channels)):
            name = 'D_z_fc' + str(i)
            current = fc(
                input_vector=current,
                num_output_length=num_hidden_layer_channels[i],
                name=name
            )
            if enable_bn:
                name = 'D_z_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
        # output layer
        name = 'D_z_fc' + str(i + 1)
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        return tf.nn.sigmoid(current), current

    def discriminator_img(self, image, y, diagnosis, is_training=True, reuse_variables=False,
                          num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = conv2d(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[i],
                size_kernel=self.size_kernel,
                name=name
            )
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
            if i == 0:
                current = concat_label(current, y)
                current = concat_label(current, diagnosis, int(self.num_progression_points / 2))
        # fully connection layer
        name = 'D_img_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name
        )
        current = lrelu(current)
        name = 'D_img_fc2'
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        # output
        return tf.nn.sigmoid(current), current

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self, model_path=None):
        if model_path is None:
            print("\n\tLoading pre-trained model ...")
            checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        else:
            print("\n\tLoading init model ...")
            checkpoint_dir = model_path
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            try:
                self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
                return True
            except Exception as e:
                print(str(e))
                return False
        else:
            return False

    def create_query_labels(self):
        labels = np.arange(self.numb_of_sample)
        labels = np.repeat(labels, self.num_progression_points)
        query_labels = np.ones(
            shape=(self.numb_of_sample * self.num_progression_points, self.num_progression_points),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]
        return query_labels

    def test(self, images, diagnosis, name):
        test_dir = os.path.join(self.save_dir, self.output_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        curr_image = images[:self.numb_of_sample, :, :, :]
        curr_diagnosis = diagnosis[:self.numb_of_sample, :]

        query_labels = self.create_query_labels()
        query_images = np.tile(curr_image, [self.num_progression_points, 1, 1, 1])
        query_diagnosis = np.tile(curr_diagnosis, [self.num_progression_points, 1])
        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: query_images,
                self.fuzzy_membership: query_labels,
                self.age: query_labels,
                self.diagnosis: query_diagnosis
            })

        save_batch_images(
            batch_images=query_images,
            save_path=os.path.join(test_dir, 'input.png'),
            image_value_range=self.image_value_range,
            size_frame=[self.num_progression_points, self.numb_of_sample]
        )
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(test_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[self.num_progression_points, self.numb_of_sample]
        )

    def normalized_regressors(self, current_region, x):
        return self.rescales[current_region].inverse_transform(self.regressors[current_region].predict_proba(x))

    def check_next_longitud_points(self, index1, index2, frames_progression, curr_diagnosis, region_loss):
        region_loss = self.compute_region_loss(index1, index2, frames_progression, curr_diagnosis, region_loss)
        return [index1, tf.add(index2, 1), frames_progression, curr_diagnosis, region_loss]

    def compute_region_loss(self, index1, index2, frames_progression, curr_diagnosis, region_loss):
        select_random_regions = np.random.permutation(self.n_of_regions)  # cannot process all the region at each iteration
        image1 = (frames_progression[index1, :, :, :] + 1) * self.ration_input_output_range
        image2 = (frames_progression[index2, :, :, :] + 1) * self.ration_input_output_range
        regional_intensity_sum1 = tf.reduce_sum(self.allMask * image1, [0, 1])
        regional_intensity_sum2 = tf.reduce_sum(self.allMask * image2, [0, 1])

        for i in range(0, self.n_regions_can_be_processed):
            current_region = select_random_regions[i]
            featureVector = tf.stack([
                tf.cast(self.bin_centers_tensor[index1], tf.float32),
                tf.cast(self.bin_centers_tensor[index2], tf.float32),
                tf.cast(curr_diagnosis[0] + 1, tf.float32)], 0)

            prediction_intensity_rate_change = tf.py_func(self.normalized_regressors, [current_region, tf.reshape(featureVector, (1, -1))], tf.float64)
            intensity_rate_change = tf.reshape((regional_intensity_sum1[current_region] + 0.1) / (regional_intensity_sum2[current_region] + 0.1), (1, 1))
            region_loss = region_loss + tf.reduce_min(
                [tf.losses.mean_squared_error(prediction_intensity_rate_change, intensity_rate_change), self.max_regional_expansion]) \
                          * tf.reduce_sum(self.allMask[:, :, current_region]) / (8 * 8)  # 8*8 is the average size of a brain region
        return region_loss

    def check_prev_longitud_points(self, index1, index2, frames_progression, curr_diagnosis, region_loss):
        region_loss = self.compute_region_loss(index1, index2, frames_progression, curr_diagnosis, region_loss)
        return [tf.add(index1, -1), index2, frames_progression, curr_diagnosis, region_loss]

    cond_1 = lambda self, index1, index2, frames_progression, curr_diagnosis, region_loss: index2 < self.num_progression_points
    cond_2 = lambda self, index1, index2, frames_progression, curr_diagnosis, region_loss: index1 >= 0

    def compute_region_regres_loss(self, frames_progression, curr_diagnosis, curr_age):
        # region loss
        region_loss = 0.0
        res = tf.while_loop(self.cond_1, self.check_next_longitud_points, [curr_age, curr_age + 1, frames_progression, curr_diagnosis, region_loss])
        res = tf.while_loop(self.cond_2, self.check_prev_longitud_points, [curr_age - 1, curr_age, frames_progression, curr_diagnosis, res[4]])
        return res[4] / self.num_progression_points

    def compute_physical_constrain(self, frames_progression, selected_query_images, curr_age, fuzzy_membership):
        # deformation loss
        pixel_reg_loss = self.voxel_based_constrain(frames_progression, selected_query_images, curr_age)
        deformation_loss = 0
        for i in range(self.num_progression_points):
            intensity_rate_changeFrame = tf.gather(frames_progression, i)
            deformation_loss = deformation_loss + tf.reduce_mean(tf.abs(intensity_rate_changeFrame - selected_query_images)) * \
                               (fuzzy_membership[i] + self.minimum_input_similarity)

        return pixel_reg_loss, deformation_loss

    def voxel_based_constrain(self, frames_progression, selected_query_images, curr_age):
        # voxel loss
        pixel_reg_loss = 0
        index1 = tf.reduce_min([curr_age, 8])
        intensity_rate_changeFrame = tf.reduce_max(tf.gather(frames_progression, [(tf.range(index1, self.num_progression_points))]), 1)
        pixel_reg_loss = pixel_reg_loss + tf.reduce_mean(tf.abs(intensity_rate_changeFrame - selected_query_images))  # L1 loss
        index2 = tf.reduce_max([curr_age, 1])
        intensity_rate_changeFrame = tf.reduce_min(tf.gather(frames_progression, [(tf.range(0, index2))]), 1)
        pixel_reg_loss = pixel_reg_loss + tf.reduce_mean(tf.abs(intensity_rate_changeFrame - selected_query_images))

        return pixel_reg_loss / 2

    def longitudinal_constrains(self, images, diagnosis, age_index, fuzzy_membership):
        query_labels=self.create_query_labels()
        data_query_labels = tf.convert_to_tensor(query_labels, np.float32)
        x = tf.constant([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

        loss_pixel_regression = tf.constant(0, dtype=tf.float32)
        loss_regional_regression = tf.constant(0, dtype=tf.float32)
        loss_deformation = tf.constant(0, dtype=tf.float32)

        perm = np.random.permutation(self.numb_of_sample - 1) + 1
        tt = perm[0]
        image = images[(tt - 1) * self.numb_of_sample:tt * self.numb_of_sample, :, :, :]
        curr_age_index = age_index[(tt - 1) * self.numb_of_sample:tt * self.numb_of_sample]
        curr_fuzzy_membership = fuzzy_membership[(tt - 1) * self.numb_of_sample:tt * self.numb_of_sample, :]
        curr_diagnosis = diagnosis[(tt - 1) * self.numb_of_sample:tt * self.numb_of_sample, :]
        query_images = tf.tile(image, tf.constant([self.num_progression_points, 1, 1, 1]))
        query_diagnosis = tf.tile(curr_diagnosis, tf.constant([self.num_progression_points, 1]))
        z = self.generator(self.encoder(query_images, True), data_query_labels, query_diagnosis, True)
        _, discriminator_result_on_sym = self.discriminator_img(z, data_query_labels, query_diagnosis, True)

        self.D_img_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_result_on_sym, labels=tf.zeros_like(discriminator_result_on_sym))
        ) / self.num_progression_points  # D_img_loss_G is num_progression_points times simpler since is applied to progression of the same input
        self.G_img_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_result_on_sym, labels=tf.ones_like(discriminator_result_on_sym))
        )

        for j in range(0, self.numb_of_sample):
            frames_progression = tf.gather(z, x + j)
            selected_query_images = tf.gather(query_images, j)
            curr_diagnosis_final = tf.where(tf.equal(curr_diagnosis[j, :], 1))[-1]
            curr_fuzzy_membership_final = curr_fuzzy_membership[j, :]
            curr_age_final = curr_age_index[j]

            c1, c2 = self.compute_physical_constrain(frames_progression, selected_query_images, curr_age_final, curr_fuzzy_membership_final)
            loss_pixel_regression = loss_pixel_regression + c1
            loss_regional_regression = loss_regional_regression + self.compute_region_regres_loss(frames_progression, curr_diagnosis_final, curr_age_final)
            loss_deformation = loss_deformation + c2

        return loss_pixel_regression / self.numb_of_sample, loss_regional_regression / (self.n_regions_can_be_processed * self.numb_of_sample), \
               loss_deformation / self.numb_of_sample

    def testing(self, testing_samples_dir, current_slice, conditioned_enabled):
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        num_samples = int(np.sqrt(self.size_batch))
        for smooth_3d in range(-2, 3):
            file_names = glob(testing_samples_dir + str(current_slice + smooth_3d) + '/*png')
            if len(file_names) < num_samples:
                print('The number of testing images is must larger than ', num_samples)
                exit(0)
            for j in range(len(file_names)):
                sample_files = file_names[j]
                sample = load_image(
                    image_path=sample_files,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                )
                if self.num_input_channels == 1:
                    images = np.array(sample).astype(np.float32)[None, :, :, None]
                else:
                    images = np.array(sample).astype(np.float32)
                images = np.repeat(images, self.num_progression_points, 0)

                diagnosis = np.ones(
                    shape=(num_samples, self.n_of_diagnosis),
                    dtype=np.float32
                ) * self.image_value_range[0]
                # curr_diagnosis = int(str(sample_files).split('/')[-1].split('_')[2]) - 1
                if conditioned_enabled:
                    for i in range(self.n_of_diagnosis):
                        diagnosis = np.ones(
                            shape=(num_samples, self.n_of_diagnosis),
                            dtype=np.float32
                        ) * self.image_value_range[0]
                        # diagnosis[:, curr_diagnosis] = self.image_value_range[-1]
                        diagnosis[:, i] = self.image_value_range[-1]
                        self.test(images, diagnosis, 'test_' + str(smooth_3d) + "_" + str(sample_files).split('/')[-1])
                else:
                    diagnosis[:, 0] = self.image_value_range[-1]
                    self.test(images, diagnosis, 'test_' + str(smooth_3d) + "_" + str(sample_files).split('/')[-1])
