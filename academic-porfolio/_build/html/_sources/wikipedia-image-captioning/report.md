# Wikipedia Image Captioning 

# Abstract

This paper will attempt to introduce a deep machine learning model that can generate and retrieve the closest caption, given an image. Image captioning is inherently a cross-disciplinary task within the sub fields of Computer Vision and Natural Language Processing (NLP), which has been a popular active area of research within the Artificial Intelligence (AI) community. A successful model deals with object detection/recognition while understanding their relative spatial properties within a given scene/location. In addition, the model should produce human-readable sentences (captions) which requires both syntactic and semantic understanding of the language. To build our image-caption deep learning model, we will utilize the  Wikipedia-based Image Text (WIT) dataset presented by the Google Research as detailed in their SIGIR published work  {cite}`wit`. This rich dataset is a massive curated set of 37.5 million image-text samples across 108 languages. This dataset is ideal as it is the largest open source dataset with 11.5 million unique images over several languages. 


# Problem Definition

The advent of the internet, social platforms and many other technological innovations have made it relatively simple to encounter large number of images in our everyday routine. Very often, these images are not paired with "alt text" and/or captions. Even some of the largest websites, i.e. Wikipedia, are susceptible to this limitation. An image-captioning model that can generate or match a description based on a given image can be utilized in automatic image indexing for content-based image retrieval tasks, and can therefore be applied to many industrial sectors including education, digital libraries, web searching, military, commerce, and bio-medicine {cite}`hossain2018comprehensive`. It is clear that open models can greatly assist and improve the accessibility and learning for all. 


# Related Work

Several approaches have been proposed in order to solve the image to caption problem. First, one approach frames the task as an \textbf{image-sentence retrieval problem} {cite}`devlin-etal-2015-language` wherein the closest caption in the image training set is retrieved and transferred over to an input test image. Alternatively, certain words in the caption may be segregated according to different features present in the training image. The caption is generated by synthesising the segregated words in the training set based on the features detected in the test image ~\cite{wu2016value}. While the aforementioned approaches have shown to achieve competitive results, the produced alt-text are limited to the words and descriptions that appear in the training set.

A second favored approach is to formulate the problem as a \textbf{caption generator task} that utilises a cascade of neural networks in a encoder-decoder architecture to construct an image description. Some {cite}`donahue2016longterm`, {cite}`vinyals2015tell`, {cite}`karpathy2015deep` utilize a \textit{inject architecture}, which employs a pretrained CNN to construct a compressed and encoded representation of an image that retains it's essential high-level features such as faces, wheels, etc. The encoded feature vectors, along with an embedded sequence of words, are subsequently fed onto a Recurrent Neural Network (RNN), such as an LSTM, in order to predict and the next word in the sequence. This model uses the RNN as a language generator. One notable variant of the inject architecture leverages the contemporary state-of-the-art technique known as transformers. This model simply replaces the LSTM with a transformer layer.

In an alternative model {cite}`kiros2014unifying`, {cite}`mao2015learning` described as the \textit{Merge Architecture}, a pretrained CNN is similarly used to encode feature vectors. However, the CNN and LSTM network respectively operate  on the image and token sequence, independently. The outputs are concatenated in a multimodal layer that interprets both outputs and subsequently fed into the sentence generator that produces the final predicted caption. Note that the distinct difference in this architecture is that the RNN is used a language model to encode word representation. 

# Methods
Our approach will utilize the Merge architecture, as shown in Figure \ref{fig:schematic}. As previously mentioned, this model utilizes a multimodal space to concatenate the image and language features. The components of this model are described in detail in the following subsections.
}

## Image Encoder
This paper utilizes the concept of transfer learning, a popular optimization method in deep learning where a model previously developed for a task is re-purposed on a second related task. It is understood that this technique not only results in faster training but can significantly improve model performance. 

This paper utilizes the Inception-v3 architecture, a computer vision model trained on over 1 million images from the ImageNet dataset for the purpose of image analysis and object detection. The InceptionV3 model consists of two primary components, a feature extraction component and a classification component. The former is a convolutional neural network consisting of symmetric and asymmetric constituents including convolutions, average pooling, max pooling, concats, and dropouts. The latter consists of fully connected multi-layer perceptrons with a softmax layer intended for classification purposes. Image \ref{fig:Inception} depicts this architecture. 


In order to encode our images into compressed encoded feature vectors, we first preprocess each image by resizing them to the required InceptionV3 input of 299 x 299 pixel, reshaping its dimensions and re-scaling all pixel values to the range of [-1, 1], sample-wise. The prepossessed images are subsequently fed into the InceptionV3 model which provides a feature vector of size 2048. The feature vector is further compressed to a 256 element vector using a fully connected neural network with a relu activation function, which is proceeded by a dropout layer to prevent overfitting. 


## Text Encoder
The sequence processor utilizes the concepts of word embedding and Long Short-Term Memory (LSTM) neural network to encode an input text sequence into a 256 element vector. Word embedding is a technique that represents words as real-valued continuous vectors in a lower-dimensional vector space such that geometrical relationships (i.e. cosine distance) between the feature vectors accurately represent the semantic relationship between words. Namely, words with similar meanings are mapped closer in vector space. LSTMs are a special archetype of Recurrent Neural Network (RNN) that is capable of handling long term dependencies, namely, this network can remember previous observations over long sequence intervals to process data.

In order to produce 256 vector element encoded representation of our sequences, the following processing steps are executed on the raw text data. 

### Text Cleaning
In order to optimize training time and performance, text cleaning is an essential step implemented in some variation for every Natural Language Processing (NLP) task. This paper preprocesses the raw alt text by applying normalization, removing unicode characters (i.e. punctuation's, emoji's, numbers, etc.) and removing one character words (i.e. I, a, etc..). Text normalization refers to converting all capital characters to lowercase so that words like "Hello" and "hello" are not interpreted differently by our model. Unicode  and one word characters do not add any descriptive value to our sentences, as such, they can be disregarded. 

### Tokenization & Padding

Deep neural networks are incapable of understanding raw text, therefore, tokenization is an essential text pre-processing technique that serves to assign an arbitrary and unique integer value to a word. Once a tokenization dictionary for a given corpus is generated, it is utilized to create vector sequences for each caption descriptions. Lastly, as all neural networks require inputs of equal length, it is important to pad each token sequence, in other words, we prepend an integer value to all token sequences such that the length of all tokenized sentences are of the same length. It is important to note that this process is distinct from word embedding as tokenization merely assigns arbitrary integer values and contain no geometrical relationships. 



## Decoder
The decoder merges the 256 element feature vectors from both the image and text encoder layers via an addition operation. This is followed a hidden dense layer with 256 neuron and final output Dense layer that produces a softmax prediction over the entire output vocabulary for the next word in the sequence.


The prediction output is generated by a fully connected neural network, as opposed to an LSTM, only one word can be generated at one each time-step during training. More specifically, if the referenced caption contains n words, the model will produce n-1 sequences, where each sequence appends an additional word. Table \ref{training_sample} provides an example of the training sequence for a given caption. 



