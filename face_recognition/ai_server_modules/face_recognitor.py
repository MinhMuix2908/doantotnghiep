
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from backbones.resnet_v1 import ResNet_v1_50
from models import MyModel
import numpy as np
from image_search import SimilaritySearch



class FaceRecognitor:
    def __init__(self, opt):
        #print("Initialize Face Recognitor")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        self.config = opt
        self.model = MyModel(ResNet_v1_50, embedding_size=self.config.embedding_size)
        ckpt = tf.train.Checkpoint(backbone=self.model.backbone)
        ckpt.restore(self.config.checkpoint).expect_partial()
        #print("Restored!")
        self.query_search = SimilaritySearch(self.config.database, self.config.filenames, self.config.threshold)
        

    def get_embeddings(self, face_image):
        """
            Get embedding vector of a face image.
            args:
                face_image: aligned face
            return:
                embbeding vector
        """
        
        image = tf.cast(face_image, tf.float32)
        image = image / 255
        image = tf.image.resize(image, (self.config['image_size'], self.config['image_size']))
        prelogits, _, _ = self.model(image, training=False)
        embeddings = tf.nn.l2_normalize(prelogits, axis=-1)
        return embeddings

    def register(self, face_image, name):
        '''
            Add a face image and name to database
        '''
        embeddings = self.get_embeddings(face_image)
        # embeddings = np.expand_dims(embeddings, axis= 0)
        self.query_search.add_vector(embeddings, name)

    def recognize(self, face_image):
        """
            Get embedding vector of a face image.
            args:
                face_image: A image face
            return:
                name
        """
        embeddings = self.get_embeddings(face_image)
        name = self.query_search.search_name(embeddings.numpy())
        return name        

if __name__ == "__main__":
    import anyconfig
    import munch
    
    opt = anyconfig.load("settings.yaml")
    opt = munch.munchify(opt)
    r = FaceRecognitor(opt)
    print(r.get_embeddings(np.ones((1,112,112,3))))