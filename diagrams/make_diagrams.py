"""Create README.md overview diagram."""

# pylint: disable=expression-not-assigned

import os

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.integration import StepFunctions
from diagrams.custom import Custom
from diagrams.gcp.analytics import Bigquery
from diagrams.gcp.storage import Storage

THIS_DIR = os.path.dirname(__file__)
ICONS_DIR = os.path.join(THIS_DIR, "icons")


with Diagram(
    filename=os.path.join(THIS_DIR, "architecture"),
    show=False,
    curvestyle="curved",
):

    storage = Storage("bucket")
    fct_player = Bigquery("fct_player")
    dim_player_last = Bigquery("dim_player_last")

    with Cluster("Scheduled GitHub Actions"):
        lgbm = Custom("lgbmranker", os.path.join(ICONS_DIR, "lgbm.png"))
        optuna = Custom("optuna", os.path.join(ICONS_DIR, "optuna.png"))
        draft = StepFunctions("draft")

        fct_player >> Edge(label="tune") >> optuna
        optuna >> Edge(label="train") >> lgbm
        lgbm >> Edge(label="simulate") >> draft
        draft >> Edge(label="dump") >> storage

    with Cluster("On-Demand\nGoogle Cloud Function"):
        lgbm = Custom("lgbmranker", os.path.join(ICONS_DIR, "lgbm.png"))
        storage >> Edge(label="load") >> lgbm
        lgbm >> Edge(label="predict") >> dim_player_last
