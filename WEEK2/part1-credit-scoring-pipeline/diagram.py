# diagram.py
from graphviz import Digraph

dot = Digraph(comment="Credit Scoring Pipeline", format="pdf")

# Nodes
dot.node("A", "Raw Data (CSV)", shape="folder")
dot.node("B", "ETL\n(data cleaning, features)", shape="box")
dot.node("C", "Model Training\n(Random Forest)", shape="box")
dot.node("D", "Evaluation\n(Accuracy, Precision, Recall, F1)", shape="box")
dot.node("E", "Flask API\n(/predict endpoint)", shape="cylinder")

# Edges
dot.edges(["AB", "BC", "CD"])
dot.edge("C", "E", label="model.pkl")

# Save & render
dot.render("diagram", format="png", view=False)

