# Speech2Face For Criminal Detection And Investigatory Evaluation

<h2> Speech Processing Model : </h2>

1. Because our model takes in the input in the format of the human voice which is perceived by the human ear, in order to process the voice 
we need time-frequency analysis.
2. We shall be using the spectral display for the audio signal processing tasks to imitate human perception termed auditory scene recognition.

 <h2> Short-Time Fourier Transform (STFT) : </h2>
 
It is a general tool we are using in the implementation of the model that is used for audio signal processing. 


<h3> scipy.signal.stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1, scaling='spectrum')      </h3>

Compute the Short-Time Fourier Transform (STFT).

1. Using the Scipy Library component STFT for signal processing, we shall try to quantify the change in a non-stationary signal’s frequency and 
phase content over time.
2. We are using a Hanning window for signal or image filtering using a fast Fourier transform. We are doing so to obtain more realistic results for the processing. We are using the length of each segment as 256.
3. First audio processing, We use up to 6 seconds of audio taken extracted from youtube. If the audio clip is shorter than 6 seconds, we repeat the audio such that it becomes at least 6 seconds long. The audio waveform is resampled at 16 kHz and only a single channel is used. 
4. Spectrograms are computed by taking STFT with a Hann window of 25 mm, a hop length of 10 ms, and 512 FFT frequency bands. Each complex spectrogram subsequently goes through the power-law compression, resulting in sgn(S)|S|0.3 for real and imaginary independently, where sgn(·) denotes the signum.
5. The STFT is one of the most frequently used tools in speech analysis and processing. It describes the evolution of frequency components over time. Like the spectrum itself, one of the benefits of STFTs is that its parameters have a physical and intuitive interpretation.
6. A further parallel with a spectrum is that the output of the STFT is complex-valued, though where the spectrum is a vector, the STFT output is a matrix. As a consequence, we cannot directly visualize the complex-valued output. Instead, STFTs are usually visualized using their log-spectra,  20log10(X(h,k)). Such 2 dimensional log-spectra can then be visualized with a heat-map known as a spectrogram.

# Voice Encoder Model (Using the Spectogram that we have generated earlier using STFT module in Librosa) :

Image classification is the process of segmenting images into different categories based on their features. 
1. A feature could be the edges in an image, the pixel intensity, the change in pixel values, and many more. 
2. An image consists of the smallest indivisible segments called pixels and every pixel has a strength often known as the pixel intensity. Whenever we study a digital image, it usually comes with three colour channels, i.e. the Red-Green-Blue channels, popularly known as the “RGB” values.
3. Now if we take multiple such images and try to label them as different individuals we can do it by analysing the pixel values and looking for patterns in them. 
However, the challenge here is that since the background, the colour scale, the clothing, etc. vary from image to image, it is hard to find patterns by analysing the pixel values alone. Hence we might require a more advanced technique that can detect these edges or find the underlying pattern of different features 
in the face using which these images can be labelled or classified. 
This is why we are using an advanced technique like CNN.

# CNN(Convolutional Neural Networks)

CNN or the convolutional neural network (CNN) is a class of deep learning neural networks. In short, CNN is a machine learning algorithm that can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to differentiate one from the other. CNN works by extracting features from the images.
# ->Steps under implementation:
 1. Conv2D
 2. MaxPooling2D
 3. Batch Normalisation
 4. Using Flatten, Dense, and Activation Functions(“Relu”)


