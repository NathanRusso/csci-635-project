import tensorflow as tf
import numpy as np
import config as cf

def make_labeled_dataset(mtg, pkm, ygo, batch_size=32,shuffle= False ):
    """
    This function takes in an train/test/validation split from each of the card games, a batch size, and bool to determine 
    if we need to shuffle the dataset.
   
    It then processes them in batches in order to apply a label for each class so we can train the model with TensorFlow

    If prompted it can shuffle the dataset so the model can get a variety of classes instead of each one sequentially.
    This can reduce potentially reduce bias in the model


    It returns a tensorflow dataset that is processed from the from_generator function
    """

    
    arrays= [mtg,pkm,ygo]
    chunk_size = batch_size//3
    chunks = [
        (class_id,start)
        for class_id, data in enumerate(arrays)
        for start in range (0,len(data),chunk_size)
    ]
    
    def batches():
        if shuffle:
            np.random.shuffle(chunks)

        for i in range(0,len(chunks),3):
            batch_imgs=[]
            batch_lbls=[]
            for class_id, start in chunks[i:i+3]:
                rows = np.asarray(arrays[class_id][start:start+batch_size])
                batch_imgs.append( rows.reshape(-1, cf.SIZE_FEATURE_HEIGHT, cf.SIZE_FEATURE_WIDTH, 1))
                batch_lbls.append(np.full(len(rows),class_id, dtype=np.int32))
            yield np.concatenate(batch_imgs, axis=0), np.concatenate(batch_lbls, axis=0)

    return tf.data.Dataset.from_generator(batches,
                                          output_signature=(
                                              tf.TensorSpec((None, cf.SIZE_FEATURE_HEIGHT, cf.SIZE_FEATURE_WIDTH, 1), tf.float32),
                                              tf.TensorSpec((None,), tf.int32)
                                          )).prefetch(tf.data.AUTOTUNE)

