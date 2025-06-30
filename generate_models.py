import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Model 1: Enhanced CNN with Residual Connections
def create_residual_model():
    inputs = layers.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    
    # Initial conv block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Residual block 1
    residual = layers.Conv2D(64, (1, 1), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])
    
    # Pooling
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Residual block 2
    residual = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])

    
    # Final layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Model 2: DenseNet-style architecture
def create_dense_model():
    def dense_block(x, filters):
        y1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        y1 = layers.BatchNormalization()(y1)
        y2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(y1)
        y2 = layers.BatchNormalization()(y2)
        return layers.concatenate([x, y2])
    
    inputs = layers.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Dense blocks
    x = dense_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = dense_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = dense_block(x, 256)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Model 3: EfficientNet-inspired
def create_efficient_model():
    inputs = layers.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    
    # Stem
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    # MBConv blocks
    def mb_conv(x, filters, expand_ratio=6):
        # Expansion
        expanded = layers.Conv2D(filters * expand_ratio, (1, 1))(x)
        expanded = layers.BatchNormalization()(expanded)
        expanded = layers.Activation('swish')(expanded)
        
        # Depthwise
        dw = layers.DepthwiseConv2D((3, 3), padding='same')(expanded)
        dw = layers.BatchNormalization()(dw)
        dw = layers.Activation('swish')(dw)
        
        # Squeeze-excite
        se = layers.GlobalAveragePooling2D()(dw)
        se = layers.Dense(filters // 4, activation='swish')(se)
        se = layers.Dense(filters * expand_ratio, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, filters * expand_ratio))(se)
        x = layers.Multiply()([dw, se])
        
        # Output
        x = layers.Conv2D(filters, (1, 1))(x)
        x = layers.BatchNormalization()(x)
        return layers.Add()([x, inputs]) if x.shape == inputs.shape else x
    
    x = mb_conv(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = mb_conv(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = mb_conv(x, 256)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training configuration
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3)
]

# Generate and save models
models_to_train = {
    "residual_cnn": create_residual_model(),
    # "dense_cnn": create_dense_model(),
    # "efficient_cnn": create_efficient_model()
}

for name, model in models_to_train.items():
    print(f"\nTraining {name}...")
    history = model.fit(
        x_train, y_train,
        epochs=50,
        validation_data=(x_test, y_test),
        batch_size=64,
        callbacks=callbacks
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} test accuracy: {test_acc:.4f}")
    
    # Save model
    filename = f"{name}_cifar10.h5"
    model.save(filename)
    print(f"Saved {filename}\n")
