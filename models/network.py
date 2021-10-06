import tensorflow as tf
from pprint import pprint
# from keras_flops import get_flops


class Network(tf.keras.Model):
    def __init__(self, config, model, _model_keys, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.model = model
        self.config = config
        self._model_keys = _model_keys

    def compile(self, optimizer, loss, run_eagerly=None):
        super(Network, self).compile(optimizer=optimizer,
                                     run_eagerly=run_eagerly,
                                     metrics=['accuracy'])
        self._loss = loss
        self.optimizer = optimizer
        # .... Define your model here ....
        # You need to have compiled your model before calling this.
        # image_inputs = tf.keras.Input(shape=(192, 320, 3), name='image_inputs')
        # preds = self.model(image_inputs, training=False)
        # fully_models = tf.keras.Model(image_inputs, preds, name='fully')
        # flops = get_flops(fully_models, batch_size=1)
        # print(f"FLOPS: {flops / 10 ** 9:.03} G")

    def train_step(self, data):
        training = True
        imgs, labels = data
        with tf.GradientTape() as tape:
            preds = self.model(imgs, training=training)

            loss = self._loss(preds, labels, self.config.train_batch_size,
                              training)
        if self.config.multi_optimizer:
            self._gradient(self.model, self.optimizer, loss['total'], tape)
        else:
            trainable_vars = self.model.trainable_variables
            grads = tape.gradient(loss['total'], trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return loss

    def test_step(self, data):
        training = False
        imgs, labels = data
        preds = self.model(imgs, training=training)
        loss = self._loss(preds, labels, self.config.test_batch_size, training)
        return loss

    def get_trainable_variables(self, model):
        def disable_training(layer):
            for l in layer.layers:
                l.trainable = False
                if hasattr(l, "layers"):
                    disable_training(l)
                elif isinstance(l, tf.keras.layers.BatchNormalization):
                    l.trainable = False
                elif isinstance(l, tf.keras.layers.Conv2D):
                    l.trainable = False

        model_dicts = self.config
        model_keys = [l.name for l in model.layers]
        model_total_vars = []
        for key in self._model_keys:
            if self.config.model_name == 'detr' and self.config.frozen_backbone and key == 'backbone':
                disable_training(model.get_layer(key))
            if key not in model_keys:
                raise KeyError('Set wrong model key')
            layer_vars = model.get_layer(key).trainable_variables
            model_dicts[key]['num_vars'] = len(layer_vars)
            model_dicts[key]['variables'] = layer_vars
            model_total_vars += layer_vars
        return model_dicts, model_total_vars

    def _gradient(self, model, optimizers, total_loss, tape):
        model_dicts, model_total_vars = self.get_trainable_variables(model)
        gradients = tape.gradient(total_loss, model_total_vars)
        for key in self._model_keys:
            num_vars = model_dicts[key]['num_vars']
            train_vars = model_dicts[key]['variables']
            self.optimizer[key].apply_gradients(
                zip(gradients[:num_vars], train_vars))
            gradients = gradients[num_vars:]

    def get_config(self):
        return super().get_config()
