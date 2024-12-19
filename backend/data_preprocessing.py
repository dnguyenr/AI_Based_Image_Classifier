import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data paths
train_dir = "/Users/dereckng/Desktop/Project/AI_Based_Image_Classifier/test_images"
val_dir = "/Users/dereckng/Desktop/Project/AI_Based_Image_Classifier/validation_data"

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)
