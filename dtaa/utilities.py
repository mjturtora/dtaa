# This file is part of dtaa.
# AUTHOR: Michael Turtora
# 11/11/2018
#

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

"""
UTILITIES.PY TOC

TOC:
generate_report() is entry point. Called by reportbuilder.py
template_fill() creates template_vars dict used to populate html template
data preprocessing and inventory functions (existence, empty columns, zeros)

table building functions (PanDAS df description tables, object names, types)
Graphics:
    missing number handling and graphics
    numeric type handling and graphics
    categorical type handling and graphics (top maxbars categories)

i/o mechanics:
argument handler
path setters
input file functions
output functions
template output
"""


import os
import time
import stat
import argparse
import shutil
from contextlib import suppress
import pandas as pd

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

from jinja2 import Environment, FileSystemLoader

pd.options.display.precision = 2
#pd.set_option('datetime_is_numeric', True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#####################################

# template_dir = '../templates'
template_dir = 'D:\Stuff\Projects\dtaa\dtaa\\templates'

env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("report_template.html")


def generate_report(nozeros, report_path, image_path, basename, df):
    """Controller function that calls I/O and plotting functions"""
    # todo: make make_output_path optional

    print(df.describe())
    #output_to_excel(df)
    # exit( 69)

    # todo: extract below to functions
    print('##########################')
    print("Beginning analysis")
    # todo: list number of empty colunms with their names...
    # todo: but AFTER describes in t..._vars so included in initial tables... or not?
    empty_columns = find_empty_columns(df)

    # initialize template list with known elements
    template_vars = template_fill(basename, nozeros, empty_columns, df)

    # after template fill to get correct initial shape
    if empty_columns.any():
        df = drop_empty_columns(empty_columns, df)

    # todo: move all plotting to plot controller function.
    # todo: add title/captions for miss figures

    # single file output plot functions, missing data plots first:

    # todo: refactor to "has_missing_values" and extract func
    if df.isnull().values.any():
        missing = True
        missnum(image_path, df)
        missmat(image_path, df)
        print('Missing values found so generating missnum plots')
        missheat(image_path, df)
        missdendro(image_path, df)
    else:
        missing = False
        print('No missing values found so no missnum plots generated')
    template_vars['missing'] = missing

    if dtype_exists(df, 'number'):
        print('Numeric data found so plotting histograms')
        plot_loghist(image_path, 'plot_loghist.png', df)
    else:
        print('No numeric data found so no log histograms plotted')

    # todo: move to template func or a multiplot function

    # plot functions with multiple file outputs need list of filenames for template
    template_vars['missbarlist'] = missbar(image_path, df)

    # todo: add logic to plot loghist_by if small number of classes in column 1
    # # IF column 1 is definitely a factor variable, plots by first column
    # template_vars['loghistbylist'] = plot_loghist_by(image_path, df)

    template_vars['logbarlist'] = plot_logbar(image_path, df)
    template_vars['topbarlist'] = plot_topbar(image_path, df)

    # todo: bar plots using pandas BY. (Don't work yet, maybe obsolete)
    #template_vars['logbarbylist'] = plot_logbar_by(df)

    html_out(report_path, basename, template.render(template_vars))
    return print('fini')


def template_fill(basename, nozeros, empty_columns, df):
    """Creates jinja2 template variable dict and populates with mandatory entries where they exist."""
    template_vars = {}
    template_vars['Window_Title'] = basename
    # todo: add file type variable to handle csv & more
    template_vars['Page_Title'] = basename + '.xlsx'
    template_vars['workbook_shape'] = df.shape
    template_vars['all_missing'] = empty_columns
    template_vars['nozeros'] = nozeros

    # generate html tables from table functions and add to template dict
    template_vars = html_to_template('columns', template_vars, df.dtypes.to_frame())

    # todo: functionalize this:
    if dtype_exists(df, 'number'):
        print('FOUND NUMERIC COLUMNS')
        template_vars = html_to_template('numeric_summary_statistics', template_vars, describe_numeric(df))
    else:
        print('NO NUMERIC COLUMNS FOUND')

    if dtype_exists(df, 'object'):
        print('FOUND OBJECT COLUMNS')
        template_vars = html_to_template('object_summary_statistics', template_vars, describe_objects(df))
    else:
        print('NO OBJECT COLUMNS FOUND')

    if dtype_exists(df, ['datetime']):
        print('FOUND DATETIME COLUMNS')
        template_vars = html_to_template('date_summary_statistics', template_vars, describe_dates(df))
    else:
        print('NO DATETIME COLUMNS FOUND')
    return template_vars


"""
Data prep and existence checks (empties and zeros)
"""


def dtype_exists(df, dtype):
    """Tests for existence of specified datatype in dataframe"""
    return len(df.select_dtypes(include=dtype).iloc[1].value_counts()) != 0


# Oddly enough, it happens:
def find_empty_columns(df):
    """Builds list of any empty columns"""
    df_described = description(df)
    return df_described[df_described['count'] == 0].index.values


def drop_empty_columns(empty_columns, df):
    """Drops empty columns from dataframe"""
    return df.drop(columns=empty_columns)


def remove_zeros(basename, df):
    """Replaces any zeros found with nan and prepends 'NOZEROS' to output fname"""
    print('Removing zeros')
    df.replace(0, np.nan, inplace=True)
    basename = 'NOZEROS ' + basename
    return basename, df

"""
PanDAS description table generation
"""

def description(df):
    """Gets PANDAS description table on all variables"""
    # transpose to get results in columns
    return df.describe(include='all', datetime_is_numeric=True).T


def describe_numeric(df):
    """Gets customized PANDAS description table on numeric variables"""
    return df.describe(percentiles=[.25, .50, .75, .90, .95, .99], include='number').T  #exclude='object').T


def describe_objects(df):
    """Gets PANDAS description table on object (character) variables"""
    return df.describe(include='object').T


def describe_dates(df):
    """Gets PANDAS description table on datetime variables"""
    return df.describe(include=['datetime'], datetime_is_numeric=True).T

"""
GRAPHICS FUNCTIONS:
"""

# third party missing data visualizations (Resident Mario missingno library).
# Condense to func(image_path, df, plot_type)
def missnum(image_path, df):
    """Makes ResidentMario's missing data barplot"""
    # https://github.com/ResidentMario/missingno
    ax = msno.bar(df)
    plt.savefig(os.path.join(image_path, 'missnum.png'), bbox_inches='tight')  # format='png', bbox_inches='tight')
    plt.close()


def missmat(image_path, df,  columns = 'All'):
    """Makes ResidentMario's missing data matrix"""
    ax = msno.matrix(df)
    plt.savefig(os.path.join(image_path, 'missmat.png'),  bbox_inches='tight')
    plt.close()


def missheat(image_path, df,  columns = 'All'):
    """Makes ResidentMario's missing data heatmap"""
    ax = msno.heatmap(df)
    plt.savefig(os.path.join(image_path, 'missheatmap.png'),  bbox_inches='tight')
    plt.close()


def missdendro(image_path, df,  columns = 'All'):
    """Makes ResidentMario's missing data dendrogram"""
    ax = msno.dendrogram(df)
    plt.savefig(os.path.join(image_path, 'missdendro.png'),  bbox_inches='tight')
    plt.close()


# grouped bar of na/zero/non-zero. Forty-four variables (columns) at a time. (maxbars)
def missbar(image_path, df):
    """Makes barplot of missing/zeros/and non-zeros grouped by variables"""
    # todo: funcionalize dimensions (needed elsewhere)
    dimensions = df.shape
    print('Dimensions = ', dimensions)
    rows = dimensions[0]
    # complex accounting of missing, zero, and non-zero data
    # start with count of non-missing AND zeros
    sr_numnotna = df.count()
    # get missing by diff
    sr_num_na = rows - sr_numnotna
    # convert 0 to missing and count non-zero values
    df.replace(0, np.nan, inplace=True)
    #todo: should be done here?
    sr_num_nonzero_values = df.count()
    # finally get zeros by diff
    sr_zeros = sr_numnotna - sr_num_nonzero_values
    df_miss = pd.DataFrame(dict(missing=sr_num_na,
                                zeros=sr_zeros,
                                non_zero_values=sr_num_nonzero_values
                                ))
    total_bars = len(df_miss)
    maxbars = 44
    missbarlist = []
    print('MISSBAR: total_bars = {}; maxbars = {}'.format(total_bars, maxbars))
    if total_bars < maxbars:
        missbarlist.append('missbar_1')
        ax = df_miss.plot.barh(figsize=(6, 13))
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, 'missbar_1' + '.png'))
        plt.close()
    else:
        start = 0
        stop = maxbars
        count = 0
        remaining_bars = total_bars
        while start <= stop:
                count += 1
                missbarlist.append('missbar_' + str(count))
                ax = df_miss[start:stop].plot.barh(figsize=(6, 13))
                # todo: functionalize plt calls
                plt.tight_layout()
                plt.savefig(make_image_path(image_path, 'missbar_' + str(count)))
                plt.close()

                start = stop  # next time through
                remaining_bars = remaining_bars - maxbars
                print('IN MISSBAR: Remaining Bars = ', remaining_bars)
                if remaining_bars >= maxbars:  # will there be a time after that
                    stop = start + maxbars
                else:
                    stop = start + remaining_bars
                print('Start= ', start, 'Stop= ', stop, '\n')
    return missbarlist

# Techniques for numeric (and time) types:
def plot_loghist(image_path, fname, df):
    """Makes log histograms of all numeric variables as subplots
    :param fname:
    """
    # todo: squawks about date if using .plot().hist() to get loglog
    # todo: missingno default figsize maybe not best?
    ax = df.hist(bins=100, log=True, xrot=45, figsize=(20, 12))  # xrot=45,
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, fname))
    plt.close()
    return


#todo: log ok for postive data, but need to check sign.
#todo: log ok for postive data, but need to check sign.


# todo: add more object techniques
# Techniques for object (text) types:

def plot_logbar(image_path, df):
    """Makes vertical log count barplots for all object columns with <  10,000 unique values"""
    # tight_layout works here, xaxis_ticks.rot() does not do well
    logbarlist = []
    for col in df.columns.values:
        if df[col].dtype == 'object':
            print(col, df[col].dtype)
            print('IN LOGBAR: Number of unique values= ', len(df[col].unique()))
            if len(df[col].unique()) < 10000:
                ax = df[col].value_counts(dropna=False).plot.bar(
                             log=True,
                             figsize=(20, 12)
                )
                col = str(col).replace('#', '-')
                logbarlist.append(col)  # KEEP AFTER REPLACE!
                plt.tight_layout()
                plt.suptitle(col)
                plt.savefig(make_image_path(image_path, 'logbar_' + col))
                plt.close()
    return logbarlist


def plot_topbar(image_path, df):
    """Makes horizontal log count barplots of top 'maxbars' unique values for all object columns"""
    maxbars = 35
    # todo: make dimensions a function or figure out how to ref tuple directly
    dimensions = df.shape
    #rows = dimensions[0]
    columns = dimensions[1]
    #numbar = columns-30
    df_tp = df.select_dtypes(include='object')
    topbarlist = []
    for col in df_tp.columns.values:
        sr_freq = df[col].value_counts(ascending=True)
        total_bars = len(sr_freq)
        tailnum = total_bars
        if tailnum > maxbars:
            tailnum = maxbars
        print('IN TOPBAR, Num Unique values= ', total_bars, '; COL NAME  = ', str(col))
        ax = sr_freq.tail(tailnum).plot.barh(figsize=(6, 10))  #figsize=(5, 18))
        plt.yticks(fontsize=7)
        plt.suptitle(str(col) + ';  Unique Values= ' + str(total_bars))  #, y=5.08)
        # https://matplotlib.org/faq/howto_faq.html#howto-subplots-adjust
        plt.subplots_adjust(left=.8)  #.7)  #, top=2)
        #plt.tight_layout() DOESN'T WORK WITH SKINNY BARS
        col = str(col).replace('#', '-')
        topbarlist.append(col)  # keep after .replace()!
        plt.savefig(make_image_path(image_path, 'topbar_' + col))
        plt.close()

        #sr_freq.iloc[columns-40:columns].plot.barh(figsize=(4, 18))
        #plt.getp(ax)  #.subplotpars)  #.update(left, bottom, right, top, wspace, hspace)
        #plt.getp(ax)
        #plt.setp(ax, ybound=[-2, 4.5])
    return topbarlist


# todo: figure out a way to plot bars BY column within pandas
# todo: THIS DOES NOT WORK! (need start:stop, it's in here somewhere)
def plot_logbar_by(image_path, df):
    """Want pandas by plots of bars with start:stop col's. Doesn't work yet!"""
    logbarlist = []
    for col in df.columns.values:
        if df[col].dtype == 'object':
            logbarlist.append(col)
            print(col, df[col].dtype)
            ax = df[col].value_counts(dropna=False).plot.bar(
                         by=df.columns.values[0],
                         log=True,
                         title=str('col'),
                         xrot=45,  #xlabelsize=6, xrot=45, ylabelsize=8,
                         figsize=(9, 9))
            plt.suptitle(col)
            #plt.tight_layout()
            #path = make_image_path(image_path, 'logbarby_plant_' + col)
            plt.savefig(make_image_path(image_path, 'logbarby_plant_' + col))
            plt.close()
    return logbarlist


"""
####################################################################
I/O
"""

def get_args(argv=None):
"""
Uses argparser to get file basename (no extension) for analysis
"""

parser = argparse.ArgumentParser(
        description="automatic exploratory data analysis on Excel file")

    # https://docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser
    parser.add_argument("source_file"
                        , help='Enter source file name without extension'
                        )

    arguments = parser.parse_args()
    source = arguments.source_file
    return source  #parser.parse_args(argv)


def make_image_path(image_path, substring):
    """Uses 'path.join' to build path strings for plots. Changes and '#'s to '-'s cuz: CSS"""
    # todo: add more html filters?
    # if substring contains '#' replace with - (just in case it hasn't already happened)
    path = os.path.join(image_path, substring.replace('#', '-') + '.png')
    return path


def make_output_path(basename):
    """Removes target folders if they exist and makes new paths"""
    # todo: make output path user settable
    # todo: make fig extension settable so .svg is easier.
    print('OUPUT PATHS:')

    # Project output paths
    report_path = os.path.join('..', 'io', 'Reports', basename)
    print(report_path)
    image_path = os.path.join('..', 'io', 'Reports', basename, 'png')
    print(image_path)

    # use as path prefix in mkdir: "Output\Report"
    print('Preparing file system')

    # # Harvey's suggestion, with mod's
    # with suppress(PermissionError, OSError):
    #     print('Damned Win 10 FILENOTFOUND exception, ignoring and moving on...')
    #     while True:
    #         with suppress(FileNotFoundError):
    #             shrmtree(report_path)
    #             print('Damned Win 10 PERMISSION exception, trying again')
    #         break

    if os.path.exists(report_path):
        def remove_readonly(func, path, _):
            """Clear the readonly bit and reattempt the removal"""
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(report_path, onerror=remove_readonly)

    while True:
        try:
            os.mkdir(report_path)
            os.mkdir(image_path)
            os.chmod(report_path, stat.S_IRWXU)
            os.chmod(image_path, stat.S_IRWXU)
            break
        except FileExistsError:
            print('Damned Win 10 FileExistsError exception on MKDIR, ignoring and moving on...')
            os.chmod(report_path, stat.S_IRWXU)
            shutil.rmtree(report_path)
            continue
        except PermissionError:
            print('Damned Win 10 PERMISSION exception on MKDIR, pause and again')
            print('sleep start')
            time.sleep(1)
            print('sleep stop')
            #os.chmod(report_path, stat.S_IRWXU)
            continue
    return report_path, image_path

"""
###########################
FILE INPUT
"""

def get_worksheet_as_df(basename):
    """
    Imports xlsx file as a dataframe.

    """
    # detect the current working directory and print it for laughs
    path = os.getcwd()
    print("The current working directory is %s" % path)
    # default extension
    extension = '.xlsx'
    path_name = os.path.join('..', 'io', basename + extension)
    #sheetname = 'reunifications'
    sheetname = 0

    try:
        print('Reading data file: "{}"'.format(path_name))
        df = pd.read_excel(path_name, sheet_name=sheetname)  #, nrows=1000)
        # todo: add csv feature someday
        # df = pd.read_csv(os.path.join(path_name))

    except FileNotFoundError:
        print('UTIL.GET_WORKSHEET_AS_DF: FileNotFound raised on INPUT')
        print('Trying again')
        df = pd.read_excel(path_name, sheet_name=sheetname)
    return df


def get_csv_as_df(basename):
    """
    Imports csv file as a dataframe.

    """
    # detect the current working directory and print it for laughs
    path = os.getcwd()
    print("IN GET_CSV: The current working directory is %s" % path)
    # default extension
    extension = '.csv'
    path_name = os.path.join('..', 'io', 'big-data-derby-2022', basename + extension)

    try:
        print('Reading data file: "{}"'.format(path_name))
        df = pd.read_csv(path_name)  #, nrows=1000)
        # todo: add csv feature someday (now maybe?!?)
        # df = pd.read_csv(os.path.join(path_name))

    except FileNotFoundError:
        print('UTIL.GET_CSV_AS_DF: FileNotFound raised on INPUT')
        print('Trying again')
        df = pd.read_csv(path_name)
    return df


"""
###########################
OUTPUTS:
"""


# Output: excel, jinja2 template, html
def output_to_excel(fname, df):
    """Outputs dataframe to hardwired excel file format. (not used)"""
    writer = pd.ExcelWriter(os.path.join('Output', fname), engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()
    return


def html_to_template(var, template_vars, df):
    """Writes a df to html format and adds to template dict with bootstrap CSS classes"""
    table_classes = 'table table-striped table-responsive'
    template_vars[var] = df.to_html(classes=table_classes)
    return template_vars


def html_out(report_path, basename, html_string):
    """Opens html file for writing report doc"""
    with open(os.path.join(report_path, basename + '.html'), 'w') as rpt:
        rpt.write(html_string)
    return

"""
################################################################
LEGACY CODE ARCHIVE, commented out to conserve namespace
"""


# def plot_loghist_by(image_path, df):
#     """Makes log histograms for all numeric variables for each 'by' variable"""
#     # todo: squawks about date if using .plot().hist() to get loglog
#     # todo: move to fuction that returns list, supply as arg?
#     loghistbylist = []
#     for col in df.columns.values:
#         if df[col].dtype != 'object':
#             loghistbylist.append(col)
#             print(col, df[col].dtype)
#             # todo: zoom (and pan?) range, bonus to choose distribution based range
#             ax = df.hist(
#                          column=col,
#                          by=df.columns.values[0],
#                          bins=100,
#                          #range=(0, 10),
#                          log=True,
#                          #xlabelsize=6, xrot=45, ylabelsize=8,
#                          xrot=45,
#                          figsize=(20, 12)
#             )
#             plt.suptitle(col)
#             #plt.tight_layout()  # tight has trouble with suptitle
#             col = str(col).replace('#', '-')
#             plt.savefig(make_image_path(image_path, 'loghistby_' + col))
#             plt.close()
#     return loghistbylist


#output_to_excel(df_upper_tri)
#pprint([["{0:0.2f}".format(j) for j in inner] for inner in upper_tri])

# # join_columns could "stack" data to "long" format
# def join_columns(column_names, start_row, stop_row, df):
#     """
#     Merges df columns into one series as strings. Subset with start:stop rows.
#     Not used anymore but may be useful.
#     """
#     first_col_name = column_names[0]
#     remaining_col_names = column_names[1:]
#     df = df.iloc[start_row:stop_row]
#     sr = df[first_col_name].str.strip()
#     for name in remaining_col_names:
#         sr += ' ' + df[name].str.strip()
#     sr = sr.unique()
#     sr = sr[~pd.isnull(sr)]
#     return sr


# def drop_zeros(df):
#     """Replaces all zeros with nan, superseded by remove_zeros()"""
#     # set to null and drop na? replace with na?
#     # use for second round of tables and graphs?
#     df.replace(0, np.nan, inplace=True)
#     return

"""
# selections used for custom project. Could be adapted to new data
def selections(df):
    print(df.describe())

    # # filter data:
    #df = df.loc[
    #          (df.loc[:, 'REQUIRED_AMT'] < 250)
    #            &
    #         (df.loc[:, 'REQUIRED_AMT'] > 0)
    #             ]

    # df = df.loc[
    #     (df.loc[:, 'REQUIRED_AMT'] > 0)
    # ]

    # df['Use_Delta'] = df['REQUIRED_AMT'] - df['QUANTITY_ISSUED']
    #
    # #df = df.loc[df.loc[:, 'Use_Delta'] > -0]
    #
    # df['Use_Error'] = df['Use_Delta']/df['REQUIRED_AMT']


    # df['Use_Error'] = df['Use_Delta'] /                \
    #                  (df['REQUIRED_AMT'] + df['QUANTITY_ISSUED'])

    #df = df.loc[df.loc[:, 'Use_Error'] > -0]
    #df.drop(['Use_Delta', 'Use_Error'], axis=1)

    # df = df.loc[df.loc[:, 'REQUIRED_AMT'] < 250]
    # df = df.loc[df.loc[:, 'QUANTITY_ISSUED'] < 500]
    # df = df.loc[df.loc[:, 'ON_HAND_QTY'] < 250000]
    # df = df.loc[df.loc[:, 'POTENTIAL_QTY'] < 100000]
    # df['DIFF Req-Iss'] = df['REQUIRED_AMT'] - df['QUANTITY_ISSUED']

    # df = df[['QUANTITY_ISSUED', 'REQUIRED_AMT', 'POTENTIAL_QTY',
    #         'AVAILABLE_QTY', 'ON_HAND_QTY']]
    # df = df.loc[df.loc[:, 'ON_HAND_QTY'] < 250000]
    # df = df.loc[df.loc[:, 'POTENTIAL_QTY'] < 100000]

    #df = df.loc[df.loc[:, 'QUANTITY_ISSUED'] < 20000]
    return df
"""