import os
import glob
from eval_vimeo90k import read_config, main, parser
parser.add_argument('--pretrain_model', type=str)
config = read_config()
args = parser.parse_args()
config['path']['pretrain_model'] = args.pretrain_model
print(config['path']['pretrain_model'])
main(config)