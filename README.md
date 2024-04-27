# iNeuronClass
    https://github.com/sunnysavita10/Indepth-GENAI/blob/main/Generative_AI_Roadmap.pptx
    https://www.youtube.com/@sunnysavita10/videos
    
Deep Learning
Types of NN
1. ANN:
    base network
2. CNN
3. RNN
4. GAN
5. Reinforcement Learning

Data: 
Structured Data:
    JSON
    CSV

Un-Structured Data
    Vision/CNN/Image Data/Computer Vision:
        Video Data
        Image classification:
            VGG
        Object Detection:
            Yolo
            SSD
        Object Segmentation:
            UNet
        OCR/Object Character Recognition/To recognize character inside an image:
            Amazon OCR

    Language Model/Text Data/Nature Language Processing/NLP:
        RNN for language model:
            GRU
            LSTM
            BIDI LSTM
            BIDI GRU
            Sequences:
        UseCases:
            Text Generation
            Text Summarization
            Q/A
            Language Translation
            Chatbot
        Audio Data:
            Audio first converted to text and processed using RNN

ANN:
    Input Layer
    Hidden Layer
    Output Layer

CNN:
Has many stages:
    1. Convolution (Here we process image)
        Image grid * kernel ==> We get convoluted image
        Kernel: 
            Is learnable parameter
        Multiply by kernel to extract features from the image
    2. Pooling:(Here we process image)
        maxpooling
        After maxpooling we get processed image
    3. Flattening:
        Convert 2d/3d to 1d layer after flatten
    4. Fully Connection

    Image is collection of number (0,255):
    1. black/white
    2. color (3 channel)
    Kernel is learnable parameter

RNN

Terminologies:
Weight
Bias
Activation Function
Loss functions
Optimizer
Forward and backward prop

Large Language Model:
    Processing sequence data
    1. Feedback Loop:
    2. ANN

Types of RNN (Recurrent/Repeated)
    1. Many to one
    2. One to Many
    3. Many to Many

RNN Architecture:
    output of previous state + input of current state processed at each step of the RNN to sustain the context.
    Encode words * Weights ==> pass to the next timestamp...
    Here we are processing the data in the sequence...  Single word and prev word at each step.

LSTM:
    RNN is used to process the sequence data.

Ways to map the data:
    1. One to Many:
        one input and many output
        Processing image throughtout the networks to genearate the output.
        Example: 
        Image Caption: One image and many output
            Geneate capture based on image
            Here we use both CNN and RNN.  CNN for input and RNN for output 
       
    2. Many to one
        many input and one output
        Example: 
        Text classification
            From text we are generating single output.
        Sentiment Analysis
        
    3. Many to Many
        many input and many output: Two ways to map the data
        Two type of many to many:
            
        Sychronize Input/Output:Same Length
            Refer to the same length: Same input length and Same output length
                NER/Name Entity Recognition
                POS

        Asynchronize Input/Output:Different Length
            Refer to the different length
                Machine Translation
                Text summarization

1. LSTM/Long Short Term Memory:
    Mapping of input to output
    Many to many architecture
    It maintaining long and term memory
    Maintain memory cells + hidden state are connected by different gates
    Forget Gates
    Input Gates
    Output Gates
    Here we are able to remember long and short term text.  Here we are processing the data in the sequence only.

2. GRU:
    Optimized LSTM architecture
    Here we have 2 gates:
        Reset Gate
        Updated Gate

RNN (1987)--> LSTM (1997)-->GRU(2014)

Encoder/Decoder 2014:
    Self Attention Mechanism
    Sequence to Sequence Learning
    Async input and output

Encoder/Decoder + Attention 2016

ULMFit/2018:
    Transfer learning

Transformers/2019:
    We are able to process large amount of data
    Encoder/Decoder + Attention 

BERT:
    Encoder Based Architecture

GPT:
    Decoder based Architecture

Encoder/Decoder
    Seq to Seq/Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
    Single context vector is passed from encoder to decoder
    Here encode was done using LSTM cells
    context vector: different length of input and output
    Not able to process longer sequences (Not more than 30 to 35 words)
    Seq/Seq:
        Used for machine translation
        Processing data in sequence
        Context vector has complete responsibility of the encoder side.

Encoder/Decoder + Attention:
    Created for language translator
    From encoder context vector is send to the decoder
    context vector has responsibility for everything.
    Every hidden state of the decoder is connected to EVERY hidden state of the encoder.  This is called the attention layer.  This attention layer is NN.
    One hidden state of decoder is connected to EVERY hidden state of the encoder.

SELF Attention

ULMFit
    https://arxiv.org/abs/1801.06146
    We can use transfer learning and fine tuning in NLP also

Transformer Paper:
    Attention is all you need

BERT Paper:
    https://arxiv.org/abs/1810.04805

GPT 1
    https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

GPT 2
    https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

GPT 3
    https://arxiv.org/abs/2005.14165




