from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import optparse
from utils.config import config
parser = optparse.OptionParser()

parser.add_option('-c', '--config', dest="config",
    help="load this config file", metavar="FILE")



print('Options', options)
print('Args:', args)


def main(args):
    print(args)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)



