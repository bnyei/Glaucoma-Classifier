from .DataProcessing import x1_train_scaled, x1_test_scaled, x2_train_scaled, x2_test_scaled, x3_train_scaled, x3_test_scaled, x4_train_scaled, x4_test_scaled 

train_datagen = ImageDataGenerator(shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True)

val_datagen = ImageDataGenerator()

train_set1 = train_datagen.flow(x1_train_scaled,
                               y1_train,
                               batch_size = 8)


val_set1 = val_datagen.flow(x1_test_scaled,
                           y1_test,
                           batch_size = 4)

train_set2 = train_datagen.flow(x2_train_scaled,
                               y2_train,
                               batch_size = 8)

val_set2 = val_datagen.flow(x2_test_scaled,
                           y2_test,
                           batch_size = 4)

train_set3 = train_datagen.flow(x3_train_scaled,
                               y3_train,
                               batch_size = 8)

val_set3 = val_datagen.flow(x3_test_scaled,
                           y3_test,
                           batch_size = 4)

train_set4 = train_datagen.flow(x4_train_scaled,
                               y4_train,
                               batch_size = 8)

val_set4 = val_datagen.flow(x4_test_scaled,
                           y4_test,
                           batch_size = 4)