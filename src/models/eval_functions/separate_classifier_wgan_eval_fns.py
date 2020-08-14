import tensorflow as tf

class EvalFunctions(object):
    """This class implements specialized operation used in the training framework"""
    def __init__(self,models):
        self.classifier = models[0]
        self.generator = models[1]
        self.discriminator = models[2]

    @tf.function
    def predict(self, x,training=True):
        """Returns a dict containing predictions e.g.{'predictions':predictions}"""
        logits_classifier = self.classifier(x[0], training=training)
        return {'predictions':tf.nn.softmax(logits_classifier,axis=-1)}

    @tf.function
    def generate(self, x,training=True):
        """Returns a dict containing fake samples e.g.{'fake_features':fake_features}"""
        fake_features = self.generator(x[1], training)
        return {'fake_features':fake_features}
    
    @tf.function
    def discriminate(self, x,training=True):
        """Returns a dict containing scores e.g.{'scores':score}"""
        scores = self.discriminator(x[0], training)
        return {'scores':scores}

    def accuracy(self,pred,y):
        correct_predictions = tf.cast(tf.equal(tf.argmax(pred,axis=-1), 
                                        tf.argmax(y[0],axis=-1)),tf.float32)
        return tf.reduce_mean(correct_predictions)
        
    @tf.function
    def compute_loss(self, x, y, training=True):
        """Has to at least return a dict containing the total loss and a prediction dict e.g.{'total_loss':total_loss},{'predictions':predictions}"""
        logits_classifier = self.classifier(x[0], training=training)
        fake_features = self.generator(x[1], training)
        true = self.discriminator(x[0], training)
        false = self.discriminator(fake_features, training)

        # Cross entropy losses
        class_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                y[0],
                logits_classifier,
                axis=-1,
            ))

        gen_loss = tf.reduce_mean((fake_features - x[1]) ** 2)
        # Wasserstein losses
        gen_loss += tf.reduce_mean(false)
        discr_loss = tf.reduce_mean(true) - tf.reduce_mean(false)

        if len(self.classifier.losses) > 0:
            weight_decay_loss = tf.add_n(self.classifier.losses)
        else:
            weight_decay_loss = 0.0

        if len(self.generator.losses) > 0:
            weight_decay_loss += tf.add_n(self.generator.losses)

        predictions = tf.nn.softmax(logits_classifier, axis=-1)

        total_loss = class_loss
        total_loss += gen_loss
        total_loss += discr_loss
        total_loss += weight_decay_loss

        return {'class_loss':class_loss, 'generator_loss':gen_loss, 'discriminator_loss':discr_loss, 'weight_decay_loss':weight_decay_loss,'total_loss':total_loss}, {'predictions':predictions, 'fake_features':fake_features}
