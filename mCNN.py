import os
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Add, Multiply, Maximum, BatchNormalization
from keras.initializers import HeNormal

def conv_block(inp, depth, kernel_size, pool_size, initializer):
    conv = Convolution2D(depth, (kernel_size, kernel_size), padding='same', activation='relu', kernel_initializer=initializer)(inp)
    batch_norm = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm)
    return pool

def createModel(height, width, depth, num_classes):
    kernel_size_1 = 7
    kernel_size_2 = 3
    pool_size = 2
    conv_depth_1 = 64  # Antes era 32
    conv_depth_2 = 32  # Antes era 16 
    drop_prob_1 = 0.25
    drop_prob_2 = 0.5
    hidden_size = 32
    initializer = HeNormal()

    inpLL = Input(shape=(height, width, depth))
    inpLH = Input(shape=(height, width, depth))
    inpHL = Input(shape=(height, width, depth))
    inpHH = Input(shape=(height, width, depth))

    # Aplicando a função conv_block para cada entrada
    pool_1_LL = conv_block(inpLL, conv_depth_1, kernel_size_1, pool_size, initializer)
    print("Dimensões após pool_1_LL:", pool_1_LL.shape)
    pool_1_LH = conv_block(inpLH, conv_depth_1, kernel_size_1, pool_size, initializer)
    print("Dimensões após pool_1_LH:", pool_1_LH.shape)
    pool_1_HL = conv_block(inpHL, conv_depth_1, kernel_size_1, pool_size, initializer)
    print("Dimensões após pool_1_HL:", pool_1_HL.shape)
    pool_1_HH = conv_block(inpHH, conv_depth_1, kernel_size_1, pool_size, initializer)
    print("Dimensões após pool_1_HH:", pool_1_HH.shape)

    #camada extra
    conv_extra_LL = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(pool_1_LL)
    batch_norm_extra_LL = BatchNormalization()(conv_extra_LL)
    pool_extra_LL = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm_extra_LL)

    conv_extra_LH = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(pool_1_LH)
    batch_norm_extra_LH = BatchNormalization()(conv_extra_LH)
    pool_extra_LH = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm_extra_LH)

    conv_extra_HL = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(pool_1_HL)
    batch_norm_extra_HL = BatchNormalization()(conv_extra_HL)
    pool_extra_HL = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm_extra_HL)

    conv_extra_HH = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(pool_1_HH)
    batch_norm_extra_HH = BatchNormalization()(conv_extra_HH)
    pool_extra_HH = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm_extra_HH)







    # Merging and further layers
    avg_LH_HL_HH = Maximum()([pool_1_LH, pool_1_HL, pool_1_HH])
    print("Dimensões após Maximum de LH, HL, HH:", avg_LH_HL_HH.shape)

    inp_merged = Multiply()([pool_1_LL, avg_LH_HL_HH])
    print("Dimensões após Multiply de LL e merged LH, HL, HH:", inp_merged.shape)



    C4 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(inp_merged)
    batch_norm_C4 = BatchNormalization()(C4)
    print("Dimensões após Convolution C4 e BatchNorm:", batch_norm_C4.shape)
    S2 = MaxPooling2D(pool_size=(4, 4))(batch_norm_C4)
    print("Dimensões após MaxPooling S2:", S2.shape)
    drop_1 = Dropout(drop_prob_1)(S2)
    print("Dimensões após Dropout 1:", drop_1.shape)

    C5 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(drop_1)
    batch_norm_C5 = BatchNormalization()(C5)
    print("Dimensões após Convolution C5 e BatchNorm:", batch_norm_C5.shape)
    S3 = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm_C5)
    print("Dimensões após MaxPooling S3:", S3.shape)

    C6 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu', kernel_initializer=initializer)(S3)
    batch_norm_C6 = BatchNormalization()(C6)
    print("Dimensões após Convolution C6 e BatchNorm:", batch_norm_C6.shape)
    S4 = MaxPooling2D(pool_size=(pool_size, pool_size))(batch_norm_C6)
    print("Dimensões após MaxPooling S4:", S4.shape)
    drop_2 = Dropout(drop_prob_1)(S4)
    print("Dimensões após Dropout 2:", drop_2.shape)

    flat = Flatten()(drop_2)
    print("Dimensões após Flatten:", flat.shape)
    hidden = Dense(hidden_size, activation='relu')(flat)
    print("Dimensões após primeira camada Densa:", hidden.shape)
    drop_3 = Dropout(drop_prob_2)(hidden)
    print("Dimensões após Dropout final:", drop_3.shape)
    out = Dense(1, activation='sigmoid')(drop_3)
    print("Dimensões antes da camada de saída:", out.shape)

    model = Model(inputs=[inpLL, inpLH, inpHL, inpHH], outputs=out)

    return model
