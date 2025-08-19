##Project Explanation

The rapid development of electric vehicles and renewable energy storage systems has intensified the demand for high-performance and cost-efficient batteries. A critical challenge in this domain lies in identifying optimal combinations of electrode and electrolyte materials, as their electrochemical stability, ionic conductivity, and compatibility directly govern battery efficiency and safety. In this work, we present an AI-driven platform for accelerated discovery of battery materials, integrating experimental electrolyte datasets with large-scale computational databases such as OQMD and QM9. Our model employs supervised learning techniques to predict key properties including ionic conductivity, stability windows, and material compatibility, enabling the ranking of candidate electrolyte–electrode pairs. The application provides an interactive interface where users can specify operating conditions (temperature, salt concentration, desired performance metrics) and receive data-driven recommendations of suitable material combinations. By unifying machine learning with domain-specific features, this platform significantly reduces the reliance on costly trial-and-error experimentation, offering a scalable pathway toward sustainable battery innovation. Future extensions of the system include the incorporation of solid-state electrolytes, active learning for dynamic dataset growth, and expansion to multi-ion chemistries such as Na- and Mg-based batteries.

##Workflow

This project builds two separate Graph Neural Networks (GNNs) — one for organic materials and one for transition metals — to predict whether each material can be used in battery applications.

Since the two datasets have different feature spaces (31 features for electrolytes, 26 for organic), we train two GNN models independently instead of forcing them into the same graph.

##

Data → Graph Conversion

We start with two tables (Pandas DataFrames):

Organics dataset: 44 feature columns + binary label (0 = not usable, 1 = usable).

Metals dataset: 21 feature columns + binary label.

Each table is turned into a graph:

Rows → nodes.

Similar nodes (via k-nearest neighbors) → edges.

Feature vectors → node features.

label column → node labels.

Function: df_to_pyg_graph()

Handles missing values (NaN → 0 by default, plus optional missingness mask).

Scales features (using StandardScaler).

Builds a k-NN adjacency graph.

Splits nodes into train/validation/test masks

##Model Definition

GCNConv layers extract structural + feature information.

ReLU activations + dropout prevent overfitting.

Final linear layer outputs logits for binary classification.

We build two models:

One for electrolytes (in_dim = 31 (+ mask)).

One for organics (in_dim = 21 (+ mask)).

GCNConv layers extract structural + feature information.

ReLU activations + dropout prevent overfitting.

Final linear layer outputs logits for binary classification.

##Training

We train each model separately:

Loss: CrossEntropyLoss.

Optimizer: Adam with weight decay.

Early stopping based on validation accuracy.

Function: train_gnn()

Trains for up to epochs, stops early if no improvement.

Tracks validation accuracy.

Reports test accuracy.

