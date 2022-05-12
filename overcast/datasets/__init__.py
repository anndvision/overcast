from overcast.datasets.jasmin import JASMIN
from overcast.datasets.jasmin import JASMINDaily

from overcast.datasets.synthetic import CATE
from overcast.datasets.synthetic import DoseResponse

DATASETS = {
    "jasmin": JASMIN,
    "jasmin-daily": JASMINDaily,
    "cate": CATE,
    "dose-response": DoseResponse,
}
