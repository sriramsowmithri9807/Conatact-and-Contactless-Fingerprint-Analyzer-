import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

def load_and_preprocess_image(image_path, target_size=(160, 160)):
    """Load and preprocess a single image."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        return img
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {str(e)}")

def create_pairs(contact_dir, contactless_dir):
    """Create pairs of contact and contactless fingerprint images."""
    if not os.path.exists(contact_dir) or not os.path.exists(contactless_dir):
        raise ValueError("Data directories do not exist. Please create data/contact and data/contactless directories.")
        
    contact_images = []
    contactless_images = []
    labels = []
    
    contact_files = [f for f in os.listdir(contact_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    contactless_files = [f for f in os.listdir(contactless_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not contact_files or not contactless_files:
        raise ValueError("No valid image files found in data directories.")
    
    print("Creating matching pairs...")
    # Create matching pairs (same person)
    for contact_file in tqdm(contact_files):
        if contact_file in contactless_files:
            contact_path = os.path.join(contact_dir, contact_file)
            contactless_path = os.path.join(contactless_dir, contact_file)
            
            try:
                contact_img = load_and_preprocess_image(contact_path)
                contactless_img = load_and_preprocess_image(contactless_path)
                
                contact_images.append(contact_img)
                contactless_images.append(contactless_img)
                labels.append(1)  # Match
            except Exception as e:
                print(f"Warning: Skipping pair due to error: {str(e)}")
    
    if not contact_images:
        raise ValueError("No valid matching pairs found. Ensure matching files exist in both directories.")
    
    print("Creating non-matching pairs...")
    # Create non-matching pairs (different people)
    for i in tqdm(range(len(contact_images))):
        idx = np.random.randint(0, len(contactless_images))
        if idx != i:
            contact_images.append(contact_images[i])
            contactless_images.append(contactless_images[idx])
            labels.append(0)  # Non-match
    
    return np.array(contact_images), np.array(contactless_images), np.array(labels)

def create_siamese_model(input_shape=(160, 160, 1)):
    """Create a Siamese network for fingerprint comparison."""
    # Base network for feature extraction
    def create_base_network(input_shape):
        input = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        return Model(input, x)
    
    base_network = create_base_network(input_shape)
    
    # Create the Siamese network
    input_contact = Input(shape=input_shape)
    input_contactless = Input(shape=input_shape)
    
    # Get encodings
    encoded_contact = base_network(input_contact)
    encoded_contactless = base_network(input_contactless)
    
    # Add a layer to compute absolute difference
    L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_contact, encoded_contactless])
    
    # Add prediction layer
    prediction = Dense(1, activation='sigmoid')(L1_distance)
    
    return Model(inputs=[input_contact, input_contactless], outputs=prediction)

def train_model():
    """Train the Siamese network."""
    try:
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/contact', exist_ok=True)
        os.makedirs('data/contactless', exist_ok=True)
        
        # Load and preprocess data
        contact_dir = 'data/contact'
        contactless_dir = 'data/contactless'
        
        print("Loading and preprocessing data...")
        contact_images, contactless_images, labels = create_pairs(contact_dir, contactless_dir)
        
        # Reshape images for CNN input
        contact_images = np.expand_dims(contact_images, axis=-1)
        contactless_images = np.expand_dims(contactless_images, axis=-1)
        
        # Split data
        print("Splitting data into train and test sets...")
        X_contact_train, X_contact_test, X_contactless_train, X_contactless_test, y_train, y_test = train_test_split(
            contact_images, contactless_images, labels, test_size=0.2, random_state=42
        )
        
        # Create and compile model
        print("Creating and compiling model...")
        model = create_siamese_model()
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("Training model...")
        batch_size = 32
        epochs = 50
        
        model.fit(
            [X_contact_train, X_contactless_train],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'models/fingerprint_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(
            [X_contact_test, X_contactless_test],
            y_test,
            verbose=1
        )
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        
        return model
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model = train_model()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        exit(1) 