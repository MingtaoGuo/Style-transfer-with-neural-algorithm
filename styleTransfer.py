import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
import scipy.io as sio
import scipy.misc as misc
from PIL import Image


def conv(inputs, w, b):
    w = tf.constant(w)
    b = tf.constant(b)
    return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b

def mapping(img):
    return 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))

class StyleTransfer:

    def __init__(self, H=256, W=256, C=3, alpha=1e-3, beta=1.0, iteration=500, content_path="./content//content.jpg", style_path="./style//style.jpg"):
        self.content_img = tf.placeholder("float", [1, H, W, C])
        self.style_img = tf.placeholder("float", [1, H, W, C])
        self.target_img = tf.get_variable("target", shape=[1, H, W, C], initializer=tf.truncated_normal_initializer(stddev=0.02))
        feature_bank_x = self.Network_vgg(self.target_img)
        feature_bank_style = self.Network_vgg(self.style_img)
        feature_bank_content = self.Network_vgg(self.content_img)
        self.L_content = self.content_loss(feature_bank_x, feature_bank_content)
        self.L_style = self.style_loss(feature_bank_x, feature_bank_style)
        self.total_loss = alpha * self.L_content + beta * self.L_style
        # self.Opt = tf.train.AdamOptimizer(0.0002).minimize(self.total_loss)
        #L-BFGS
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.total_loss, method='L-BFGS-B',options={'maxiter': iteration, 'disp': 0})
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.train(H, W, C, content_path, style_path)

    def train(self, H, W, C, content_path, style_path):
        content_img = np.reshape(misc.imresize(np.array(Image.open(content_path)), [H, W], mode="RGB"), [1, H, W, C])
        style_img = np.reshape(misc.imresize(np.array(Image.open(style_path)), [H, W], mode="RGB"), [1, H, W, C])
        self.sess.run(tf.assign(self.target_img, content_img), feed_dict={self.content_img: content_img, self.style_img: style_img})
        self.optimizer.minimize(self.sess, feed_dict={self.content_img: content_img, self.style_img: style_img})
        L_content = self.sess.run(self.L_content, feed_dict={self.content_img: content_img, self.style_img: style_img})
        L_style = self.sess.run(self.L_style, feed_dict={self.content_img: content_img, self.style_img: style_img})
        L_total = self.sess.run(self.total_loss, feed_dict={self.content_img: content_img, self.style_img: style_img})
        print("L_content: %g, L_style: %g, L_total: %g" % (L_content, L_style, L_total))
        target_img = self.sess.run(self.target_img,feed_dict={self.content_img: content_img, self.style_img: style_img})
        Image.fromarray(np.uint8(mapping(np.reshape(target_img, [H, W, C])))).save("./deepdream/target.jpg")




    def content_loss(self, feature_bank_x, feature_bank_content):
        #content loss
        #squared-error
        return tf.reduce_sum(tf.square(feature_bank_x["relu4_2"] - feature_bank_content["relu4_2"])) / 2.0

    def style_loss(self, feature_bank_x, feature_bank_style):
        #style loss
        E = 0
        for layer in feature_bank_style.keys():
            if layer == "relu1_1" or layer=="relu2_1" or layer=="relu3_1" or layer=="relu4_1" or layer=="relu5_1":
                w = 0.2
            else:
                w = 0
            C = int(feature_bank_x[layer].shape[-1])
            H = int(feature_bank_x[layer].shape[1])
            W = int(feature_bank_x[layer].shape[2])
            F = tf.reshape(tf.transpose(feature_bank_x[layer], [0, 3, 1, 2]), shape=[C, -1])
            #Gram matrix of x
            G_x = tf.matmul(F, tf.transpose(F))
            C = int(feature_bank_style[layer].shape[-1])
            F = tf.reshape(tf.transpose(feature_bank_style[layer], [0, 3, 1, 2]), shape=[C, -1])
            #Gram matrix of style
            G_s = tf.matmul(F, tf.transpose(F))
            E += w * tf.reduce_sum(tf.square(G_x - G_s)) / (4 * C**2 * H**2 * W**2)
        return E

    def Network_vgg(self, inputs):
        vgg_para = sio.loadmat("./vgg_para//vgg.mat")
        layers = vgg_para["layers"]
        feature_bank = {}
        with tf.variable_scope("vgg"):
            for i in range(37):
                if layers[0, i][0, 0]["type"] == "conv":
                    w = layers[0, i][0, 0]["weights"][0, 0]
                    b = layers[0, i][0, 0]["weights"][0, 1]
                    with tf.variable_scope(str(i)):
                        inputs = conv(inputs, w, b)
                elif layers[0, i][0, 0]["type"] == "relu":
                    inputs = tf.nn.relu(inputs)
                    feature_bank[layers[0, i][0, 0]["name"][0]] = inputs
                else:
                    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        return feature_bank

if __name__ == "__main__":
    st = StyleTransfer(H=512, W=512, C=3, alpha=1e-5, beta=1.0, iteration=500, content_path="./content//content.jpg", style_path="./style//style.jpg")