High Performance Computing (HPC)
________________________________________
Assignment 1: Parallel BFS and DFS using OpenMP
1.	Define BFS and DFS in graph traversal.
2.	What are the time complexities of BFS and DFS?
3.	How does parallelism improve BFS/DFS performance?
4.	How is a visited array managed safely in parallel BFS?
5.	What are critical sections in OpenMP?
6.	What issues arise when parallelizing DFS?
7.	What is breadth-wise parallelism in BFS?
8.	What happens if synchronization is not handled in parallel traversal?
9.	What OpenMP directive is used to parallelize a for loop?
10.	What are advantages and disadvantages of parallel BFS?
________________________________________
Assignment 2: Parallel Bubble Sort and Merge Sort using OpenMP
1.	Why is Merge Sort called a "divide and conquer" algorithm?
2.	What happens if two threads try to swap the same elements in parallel Bubble Sort?
3.	How do you divide the array for parallel Merge Sort?
4.	How does OpenMP manage workload among threads?
5.	What is the significance of choosing correct grain size?
6.	What is the worst-case time complexity of parallel Bubble Sort?
7.	How does thread scheduling affect performance in sorting?
8.	What is false sharing in parallel sorting?
9.	What is memory overhead in Merge Sort and how do you manage it?
10.	How would you optimize a parallel sorting algorithm for large datasets?
________________________________________
Assignment 3: Parallel Reduction Operations using OpenMP
1.	What is associative operation? Why is it important for reduction?
2.	What does the reduction clause in OpenMP do internally?
3.	How do you perform custom reduction operations in OpenMP?
4.	Why might a parallel reduction be slower than serial for small datasets?
5.	Explain the performance impact of cache coherence during reduction.
6.	What are atomic operations and how do they differ from reduction?
7.	How can thread-local variables help in reduction?
8.	How would you reduce communication overhead during parallel reduction?
9.	In what scenarios is reduction critical for scientific computing?
10.	What are the common pitfalls when writing reduction code in OpenMP?
________________________________________
Assignment 4: CUDA Programming: Vector Addition and Matrix Multiplication
1.	What is the structure of a CUDA program?
2.	What is a CUDA kernel launch configuration syntax?
3.	How is memory allocated on GPU?
4.	What are global, shared, and local memories in CUDA?
5.	How is thread indexing done in a CUDA kernel?
6.	What happens if too many threads are launched in CUDA?
7.	What is warp size in CUDA?
8.	How does coalesced memory access improve performance?
9.	What are CUDA streams and why are they useful?
10.	Explain how grid-stride loops help in CUDA programming.
________________________________________
Assignment 5: Mini Project (HPC)
1.	What problem statement did you address in your mini-project?
2.	What parallel programming techniques did you apply?
3.	Which OpenMP/CUDA constructs were most helpful?
4.	How did you handle load balancing?
5.	What profiling tools did you use to measure performance?
6.	How did you optimize memory usage in your project?
7.	What results did you achieve and how did they compare to the sequential approach?
8.	What were the main bottlenecks you identified?
9.	If you had more time, what would you improve in your project?
10.	How does your project contribute to real-world HPC problems?
 	
1.	What are the applications of Parallel Computing. 
2.	2. What is the basic working principle of VLIW Processor. 
3.	3. Explain control structure of Parallel platform in details. 
4.	4. Explain basic working principle of Superscalar Processor. 
5.	5. What are the limitation of Memory System Performance. 
6.	6. Explain SIMD, MIMD & SIMT Architecture.
7.	7. What are the types of Dataflow Execution model. 
8.	8. Write a short notes on UMA, NUMA & Level of parallelism. 
9.	9. Explain cache coherence in multiprocessor system. 
10.	10. Explain N-wide Superscalar Architecture. 
11.	11. Explain interconnection network with its type? 
12.	12. Write a short note on Communication Cost In Parallel machine. 
13.	13. Compare between Write Invalidate and Write Update protocol. 
14.	1. Explain decoposition, Task & Depedancy graph.
15.	2. Explain Granularity, Concurrency & Task interaction.
16.	3. Explain decoposition techniques with its types. 
17.	4. What are the characteristics of Task and Interactions? 
18.	5. Explain the Mapping techniques in details. 
19.	6. Explain parallel Algortithm Model. 
20.	7. Explain Thread Organization. 
21.	8. Write a short note on IBM CBE 
22.	9. Explain hisory of GPUs and NVIDIA Tesla GPU. 
23.	1. Explain Broadcast & Reduce operation with help of diagram. 
24.	2. Explain One-to-all broadcast and reduction on a Ring? 
25.	3. Explain Operation of All to one broadcast & Reduction on a ring? 
26.	4. Write a pseudo code for One-to-all broadcast alogrithm on hypercube with different cases? 
27.	5. Explain term of All-to-all broadcast & reduction on Liner array, mesh and Hypercube topologies. 
28.	6. Explain Scatter and Gather Operation.
29.	7. Write short note on Circular shaft on Mesh and hypercube. 
30.	8. Explain different approaches of Communication operation. 
31.	9. Explain all to all personalized communication
 	🤖 Deep Learning (DL)
________________________________________
Assignment 1: Linear Regression using Deep Neural Networks
32.	What is the architecture of your DNN model for regression?
33.	Why is a deep neural network sometimes used instead of simple linear regression?
34.	What are the advantages of using ReLU over Sigmoid in hidden layers?
35.	What initialization technique was used for weights?
36.	What happens if learning rate is too high or too low?
37.	What is the role of batch size during training?
38.	How does regularization help in linear regression models?
39.	What is the effect of too many hidden layers on a simple regression problem?
40.	How do you validate your model's performance?
41.	What are the risks of underfitting in linear regression?
________________________________________
Assignment 2: Classification using Deep Neural Networks
1.	What is the architecture of your classification DNN?
2.	How do you encode class labels for multi-class classification?
3.	What is softmax activation and how does it work?
4.	What is categorical cross-entropy loss?
5.	How can dropout help during training?
6.	How do you handle overfitting in classification models?
7.	What is the confusion matrix and why is it important?
8.	How do you deal with unbalanced datasets?
9.	What is data augmentation and how is it used in classification tasks?
10.	What are learning rate schedules and why are they important?
________________________________________
Assignment 3: CNN for Fashion MNIST Classification
1.	Why are CNNs better for images than fully connected networks?
2.	What is the size of the filter you used and why?
3.	What is feature map in CNN?
4.	Explain the concept of receptive field in CNN.
5.	How does a CNN handle translation invariance in images?
6.	Why is max pooling preferred over average pooling?
7.	What optimizer did you choose and why?
8.	How do vanishing gradients affect CNNs?
9.	What are Batch Normalization layers and how do they help?
10.	How did you tune hyperparameters (like learning rate, epochs)?
________________________________________
Assignment 4: Mini Project - Human Face Recognition
1.	What preprocessing steps did you apply to the images?
2.	What CNN architecture or model did you use (custom or pre-trained)?
3.	What is face embedding in deep learning?
4.	How does Triplet Loss function work in face recognition?
5.	How do you differentiate between classification and verification tasks in face recognition?
6.	How did you handle pose and lighting variations?
7.	What is one-shot learning and where is it useful in face recognition?
8.	What data augmentation techniques helped your project?
9.	How would you deploy the face recognition system into a mobile app?
10.	What improvements can be done using transfer learning?
11.	What is Batch Size? 
12.	2. What is Dropout? 
13.	3. What is RMSprop? 
14.	4. What is the Softmax Function? 
15.	5. What is the Relu Function?

16.	What is Binary Classification?
17.	 2. What is binary Cross Entropy? 
18.	3. What is Validation Split? 
19.	4. What is the Epoch Cycle? 
20.	5. What is Adam Optimizer?

21.	What is Linear Regression?
22.	2. What is a Deep Neural Network?
23.	3. What is the concept of standardization?
24.	4. Why split data into train and test?
25.	5. Write Down Application of Deep Neural Network?

26.	What is MNIST dataset for classification? 
27.	2. How many classes are in the MNIST dataset?
28.	 3. What is 784 in MNIST dataset? 
29.	4. How many epochs are there in MNIST? 
30.	5. What are the hardest digits in MNIST?
31.	What do you mean by Exploratory Analysis? 
32.	2. What do you mean by Correlation Matrix? 
33.	3. What is Conv2D used for
