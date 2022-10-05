"""Create README.md overview diagram."""

# pylint: disable=expression-not-assigned

import os

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.network import APIGateway
from diagrams.custom import Custom
from diagrams.gcp.analytics import Bigquery
from diagrams.gcp.compute import Functions
from diagrams.gcp.storage import Storage
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.vcs import Git, Github
from diagrams.programming.language import Python


THIS_DIR = os.path.dirname(__file__)
ICONS_DIR = os.path.join(THIS_DIR, "icons")


with Diagram(
    filename=os.path.join(THIS_DIR, "overview"),
    show=False,
    curvestyle="curved",
):

    storage = Storage("bucket")
    fct_player = Bigquery("fct_player")
    dim_player_last = Bigquery("dim_player_last")

    with Cluster("Scheduled GitHub Actions"):
        lgbm = Custom("lgbmranker", os.path.join(ICONS_DIR, "lgbm.png"))
        optuna = Custom("optuna", os.path.join(ICONS_DIR, "optuna.png"))
        draft = APIGateway("draft api")

        fct_player >> Edge(label="tune") >> optuna
        optuna >> Edge(label="train") >> lgbm
        lgbm >> Edge(label="simulate") >> draft
        draft >> Edge(label="dump") >> storage

    with Cluster("On-Demand\nGoogle Cloud Function"):
        lgbm = Custom("lgbmranker", os.path.join(ICONS_DIR, "lgbm.png"))
        storage >> Edge(label="load") >> lgbm
        lgbm >> Edge(label="predict") >> dim_player_last


with Diagram(
    "\nFlow",
    filename=os.path.join(THIS_DIR, "flow"),
    show=False,
    curvestyle="curved",
):
    gh_pr = Github("pull request to main")
    gh_merge = Github("merge to main")

    Git("push to dev") >> gh_pr
    gh_pr >> GithubActions("testing") >> Python("pytest") >> gh_merge
    gh_pr >> GithubActions("linting") >> Python("pylint") >> gh_merge
    gh_merge >> GithubActions("deploy") >> Functions("Deploy")
    gh_merge >> GithubActions("train") >> Storage("Dump")
