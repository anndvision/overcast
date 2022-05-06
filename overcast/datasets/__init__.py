from overcast.datasets.jasmin import JASMIN
from overcast.datasets.jasmin import JASMINDaily
from overcast.datasets.synthetic import Synthetic

DATASETS = {
    "jasmin": JASMIN,
    "jasmin-daily": JASMINDaily,
    "synthetic": Synthetic,
}
