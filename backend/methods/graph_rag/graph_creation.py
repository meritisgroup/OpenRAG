import igraph as ig  # Pour créer des graphs
import leidenalg as la  # Pour trouver des clusters dans un graph
from typing import Union
from ...database.rag_classes import Relation, MergeEntityDocument, MergeEntityOverall
import networkx as nx
from pyvis.network import Network
import numpy as np
import warnings


class Weight:
    def __init__(
        self,
        relation: Relation,
        relations: list[Relation],
        entities: list[Union[MergeEntityDocument, MergeEntityOverall]],
    ):
        self.relation = relation

        source_degree, target_degree = 1, 1

        for rel in relations:

            if rel != relation:

                if rel.source == relation.source or rel.target == relation.source:
                    source_degree += 1

                if rel.source == relation.target or rel.target == relation.target:
                    target_degree += 1

        for entity in entities:

            if entity.name == relation.source:
                source_degree += entity.degree

            if entity.name == relation.target:
                target_degree += entity.degree

        self.source_degree = source_degree
        self.target_degree = target_degree

    def get_weight(self):
        """
        Weight of a relation in knowledge graph is sum of node's degrees
        """
        return self.source_degree + self.target_degree


class Graph:
    def __init__(
        self,
        entities: list[Union[MergeEntityDocument, MergeEntityOverall]],
        relations: list[Relation],
    ) -> None:
        """ """
        entities_names = [entity.name for entity in entities]
        cleaned_relations = [
            relation
            for relation in relations
            if relation.source in entities_names and relation.target in entities_names
        ]

        edges = []
        for relation in cleaned_relations:

            source_name = relation.source
            target_name = relation.target

            for entity in entities:

                if entity.name == source_name:
                    source_id = entity.id

                if entity.name == target_name:
                    target_id = entity.id

            edges.append((source_id, target_id))
            edges.append((target_id, source_id))

        self.edges = edges

        directional_weights = [
            Weight(relation, cleaned_relations, entities).get_weight()
            for relation in cleaned_relations
        ]
        weights = []

        for w in directional_weights:
            weights.append(w)
            weights.append(w)

        self.weights = weights

        graph = ig.Graph(self.edges)
        graph.es["weight"] = self.weights

        self.graph = graph
        self.entities = entities
        self.communities = []
        # self.colors = list(ig.drawing.colors.known_colors.keys())

    def show_properties(self):

        res = ""

        print("\n\n")
        number_of_nodes = self.graph.vcount()
        print(f"Nombre de noeuds : {number_of_nodes}")
        res += f"Nombre de noeuds : {number_of_nodes}" + "\n"

        degrees = self.graph.degree()
        degree_mean = sum(degrees) / len(degrees)
        print(f"Degré moyen des noeuds : {degree_mean:.2f}")
        res += f"Degré moyen des noeuds : {degree_mean:.2f}" + "\n"

        maximal_degree = max(degrees)
        maximal_degree_node = degrees.index(maximal_degree)
        print(f"ID du noeud de degré maximal : {maximal_degree_node}")
        res += f"ID du noeud de degré maximal : {maximal_degree_node}" + "\n"

        print(f"Degré du noeud de degré maximal : {maximal_degree}\n")
        res += f"Degré du noeud de degré maximal : {maximal_degree}\n"

        return res

    def create_communities(self) -> list[list[int]]:
        """
        Clusterise the graph and returns a list of lists containing entities' ids
        """
        partitions = la.find_partition(
            self.graph,
            la.CPMVertexPartition,
            resolution_parameter=0.1,
            weights=self.weights,
            n_iterations=5,
        )

        communities = []
        for i, community in enumerate(partitions):
            communities.append(community)

        communities_filtered = [
            community for community in communities if len(community) > 1
        ]

        self.communities = communities_filtered

        return communities_filtered

    def retrieve_entity_name_from_id(
        self, entities: list[Union[MergeEntityDocument, MergeEntityOverall]], id: int
    ) -> str:
        """
        Returns the name of en entity using its ids and a list of entities
        """
        for entity in entities:

            if entity.id == id:
                return entity.name

        return ""

    def retrieve_community_from_id(self, id: int) -> str:
        """
        Returns the id of the entity's community using its ids
        """
        if self.communities == []:
            print("Communities haven't been calculated.")
            return -1

        for k, sub_community in enumerate(self.communities):

            for belonging_id in sub_community:

                if id == belonging_id:
                    return k

        return -1

    def convert_to_nx(self):

        G = nx.Graph()

        colors = [
            f"rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})"
            for _ in range(len(self.communities) + 1)
        ]

        for entity in self.entities:

            label = entity.name
            title = entity.description
            color = colors[self.retrieve_community_from_id(entity.id)]

            G.add_node(label, title=title, color=color)

        for couple_id, weight in zip(self.edges, self.weights):

            source = self.retrieve_entity_name_from_id(self.entities, couple_id[0])
            target = self.retrieve_entity_name_from_id(self.entities, couple_id[1])
            title = str(weight)

            G.add_edge(source, target, title=title)

        return G

    def plot_graph(self, storage_path: str, graph_name: str) -> None:
        """
        Create a html file that you can localy execute to play with the graph
        """
        net = Network(
            notebook=False,
            height="600px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
        )

        net.from_nx(self.convert_to_nx())

        net.show_buttons(filter_=["physics"])
        net.repulsion(node_distance=100)

        if storage_path[-1] != "/":
            storage_path += "/"

        warnings.filterwarnings(
            "ignore",
            message="Warning: When cdn_resources is 'local' jupyter notebook has issues displaying graphics on chrome/safari. Use cdn_resources='in_line' or cdn_resources='remote' if you have issues viewing graphics in a notebook.",
        )
        net.generate_html(
            name=f"{storage_path}{graph_name}.html", local=True, notebook=False
        )
        net.save_graph(f"./{storage_path}{graph_name}.html")
