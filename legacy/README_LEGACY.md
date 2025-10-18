# Car-DCGAN-Keras

![training](training.gif)
![exemple](gen/1554321002.6462195_1.0.png)
![exemple2](gen/1554321005.7100232_1.0.png)
![exemple3](gen/1554321005.2707386_1.0.png)
![exemple3](gen/1554321004.0519686_1.0.png)
![exemple4](gen/1554321002.9354467_1.0.png)

* My implementation of a DCGAN in keras with tensorflow.

* The usage is simple, run gan.py to train the generator and discriminator, if you want to generate images, just execute generate.py.
You can create a gif from your training image by executing gif.py.

* training image are in the "car" folder, generated images from generate.py are in the "gen" folder (some exemples are already in the folder), the dataset should be in the "car_img" folder (a few images are in just for exemple, if you want the full dataset, please download : http://ai.stanford.edu/~jkrause/cars/car_dataset.html). Models are in the "vroum" folder (vroumdis.h5 for the discriminator and vroumgen.h5 for the generator).

