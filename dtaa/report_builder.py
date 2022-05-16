# This file is part of dtaa.
# AUTHOR: Michael Turtora
# 11/11/2018

# dtaa is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# dtaa is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with dtaa .  If not, see <https://www.gnu.org/licenses/>.

import time
import utilities as util

def build_report(basename):

    # activate this to run report
    nozeros = False  # True removes zeros and makes separate output

    df = util.get_worksheet_as_df(basename)
    if nozeros:
        basename, df = util.remove_zeros(basename, df)
    else:
        print('Keeping zeros')    #    report_path, image_path = make_output_path(test, basename)
    basename = basename  # + ' GROUPED'
    report_path, image_path = util.make_output_path(basename)
    util.generate_report(nozeros, report_path, image_path, basename, df)

    # activate this to load data w/o running report
    #report_path, image_path, df = util.choose_file()


# import timeit
# print("{0:0.3f}".format(timeit.timeit("generate_report()",
#                     setup="from utilities import generate_report",
#                     number=1)))

if __name__ == '__main__':
    # basename = 'work_orders_routine'
    # basename = 'x sample test address city state zip 21804 addresses'
    #basename = 'Reunification Re-entry Project'
    #basename = 'Reunifications 2017-18 -no case ID'
    #basename = 're-entry cleaned sheet 0'
    basename = util.get_args()
    print(f'Called get_args and got: \n {basename} \n')
    build_report(basename)
