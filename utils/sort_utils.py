# -*- coding: utf-8 -*-
"""自然排序：正确处理文件名中的数字 (EF_24 < EF_100)"""

import re


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]
