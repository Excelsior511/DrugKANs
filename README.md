## DrugKANs: A Paradigm to Enhance Drug-Target Interaction Prediction with KANs ##

**Abstract** Identifying potential drug-target interactions (DTIs) is crucial for understanding drug mechanisms, and recent computational methods have yielded promising re sults in this area. However, these methods face several challenges, including limited model generalization due to heavy reliance on multiple similarity datasets and complex feature extraction, as well as a lack of interpretability by ignoring intrinsic information about drugs and targets. To address these challenges, we propose DrugKANs, a novel DTI prediction model that enhances both the quality and interpretability of DTI representations by integrating a dual tower architecture with Kolmogorov-Arnold Network (KAN) technology. Our model involves utilizing a pre-trained model to derive initial representations of drugs and targets, and employing a lightweight attention mechanism to cap ture key features, thereby improving representation quality. We leverage the dual-tower architecture and a lightweight feature interaction mechanism to extract high-level repre sentations separately for drugs and targets, aiming to re duce complex feature interactions and mitigate overfitting. Additionally, we incorporate a contrastive learning strategywithin the drug-target bipartite graph to address sparse neighborhood effects and enhance topological information. The inclusion of KAN technology further improves the inter pretability of the DTI prediction model. Experimental results on public datasets demonstrate that our model predicts DTIs effectively, underscoring its potential as a valuable tool in drug discovery. This comprehensive methodology presents a balanced approach to overcoming the identified challenges in DTI prediction.

### Requirements ###
python = 3.9

torch = 1.12.0

torch-geometric = 2.3.1

torch-sparse = 0.6.15

tensorflow = 2.9.1

transformers = 4.34.1

scikit-learn = 1.2.2

numpy = 1.23

pandas = 1.3.5

tqdm = 4.65.0

You can install using pip like:
```
pip install torch==1.12.0
```

### Run ###
Make sure your code flie like:

- DrugKANs
	- dataset
	  - BindingDB
	  - davis
	  - YAM
		  - enzyme
		  - gpcr
  - kan.py
  - main.py
  - model.py
  - utils.py

Then you can run the model:
```
main.py
```
