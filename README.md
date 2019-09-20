# Attention-based-Graph-CNN-for-scene-classification

This set of codes implements multi-label classification of images using graph convolution network (GCN). In this work, it is implemented for VHR airborne images using the <a href="http://bigearth.eu/assets/docs/multilabels.zip">multi-label annotated UCMERCED dataset</a>  (source: <a href="https://ieeexplore.ieee.org/document/8089668/">Chaudhuri et al</a>) but it is a generic framework. The GCN used in this framework is inpired from <a href="https://ieeexplore.ieee.org/document/7979525/">Such et al</a>. The codes are written using the TensorFlow framework (version 1.2.1) in Python 2.7.12. The input needed for the code is adjacency matrices of the graph, node features and label set. 

To implement the code:

run the <b>run_graph.py</b> file (for terminal based, type ‘python run_graph.py’ in terminal)

