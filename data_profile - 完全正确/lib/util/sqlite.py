'''
Basic practical functions for quickly sql command using
Created on April 20, 2016
@author: Hong Hao
@email: omeganju@gmail.com
'''

import sqlite3
import numpy as np


def sql_tuple(data_array, warp=True):
    '''generates a tuple for format printing as a sql query command
    '''
    for i, item in enumerate(data_array):
        if isinstance(item, str):
            item = item.strip('"')
            if warp:
                data_array[i] = '"' + str(item) + '"'
            else:
                data_array[i] = str(item)
    return tuple(data_array)

def format_str(length, style='parens'):
    '''generates a specified length of non-formatting string with warps or not
    '''
    warps = {'parens':['(',')'], 'brackets':['[',']'], 'braces':['{','}']}
    frt_str = ", ".join(["%s"] * length)
    if style:
        frt_str = frt_str.join(warps[style])
    return frt_str

def query_str(query):
    '''converts a dict type input <query> to the conditions part of a WHERE clause in a sql command'''
    conditions = []
    for column, scope in query.items():
        if isinstance(scope,str):
            str_ = '{0} = "{1}"'.format(column, scope.strip('"'))
            conditions.append(str_)
        else:
            str_ = "%s IN "%column + format_str(len(scope)) % sql_tuple(scope)
            conditions.append(str_)
    query_cmd = " WHERE " + " AND ".join(conditions)
    return query_cmd

def add_table_to_db(conn, table_name, headers):
    '''creates a table with headers to database
    '''
    # includes table_name and headers into a valid SQL command
    command_str = "CREATE TABLE %s" % table_name
    command_str += format_str(len(headers)) % sql_tuple(headers)
    # connects to the database and create the table
    cur = conn.cursor()
    cur.execute(command_str)
    conn.commit()
    cur.close()

def add_row_to_table(conn, table_name, data_array, single=True):
    '''adds the specified array of data to the desired table
    '''
    cur = conn.cursor()
    if single:
        # includes table_name and data_array into a valid SQL command
        command_str = "INSERT INTO %s VALUES " % table_name + \
            format_str(len(data_array)) % sql_tuple(data_array)
        # connects to the database and add data_array to the table
        cur.execute(command_str)
    else:
        length = len(data_array[0])
        for each in data_array:
            command_str = "INSERT INTO %s VALUES " % table_name + \
                format_str(length) % sql_tuple(each)
            cur.execute(command_str)
    conn.commit()
    cur.close()

def info(conn, verbose=False):
    '''retrieves connected database infomation including table names and corresponded column headers and dimensions
    '''
    cmd_table_names = """SELECT name FROM sqlite_master WHERE type='table'"""
    cur = conn.cursor()
    cur.execute(cmd_table_names)
    table_names = [str(name[0]) for name in cur.fetchall()]

    tables_info = {}

    for name in table_names:
        table_info = {}
        cur.execute("PRAGMA table_info(%s)" % name)
        table_headers = [str(header[1]) for header in cur.fetchall()]
        table_info['headers'] = table_headers
        col_num = len(table_headers)
        cur.execute('SELECT %s FROM %s' % (table_headers[0], name))
        row_num = len(cur.fetchall())
        dimension = tuple([col_num, row_num])
        table_info['dimension'] = dimension
        tables_info[name] = table_info
    cur.close()

    if verbose:
        if len(tables_info) == 1:
            print("DB_INFO: One table in connected Database!")
        else:
            print("DB_INFO: %s tables in connected Database!" % len(tables_info))
        for name in tables_info:
            print("=" * 80)
            print("| Table:   %s" % name)
            print("| Headers: " + format_str(col_num) % sql_tuple(tables_info[name]['headers'], warp=False))
            print("| Dim:     %s x %s" % tables_info[name]['dimension'])
        print("=" * 80)
    return tables_info

def freq(conn, table, column, top=9, verbose=True, order='DESC', **query):
    '''Counts the frequency of column in the table with the condition claimed by query
    '''
    cmd_first = "SELECT {0}, count(*) FROM {1}".format(column, table)
    if query:
        cmd_middle = query_str(query)
    else:
        cmd_middle = ""
    cmd_last = " GROUP BY {0} ORDER BY count(*) {1}".format(column,order)
    cmd = cmd_first + cmd_middle + cmd_last

    cur = conn.cursor()
    print(cmd)
    cur.execute(cmd)
    frequency = [(str(x[0]), str(x[1])) for x in cur.fetchall()[:top]]
    cur.close()

    if verbose:
        queried_len = len(frequency)
        print("FREQ: frequency of {0} in '{1}' (top {2} listed)".format(column, table, queried_len))
        print("=" * 80)
        print("Limitation: " + cmd_middle.strip() + " ({0})".format(order))
        for ind in range(0, queried_len, 3):
            prt_list = frequency[ind:ind+3]
            prt_str = ", ".join(["%s = %s"] * len(prt_list))
            prt_tpe = tuple(np.ravel(prt_list))
            if ind == 0:
                print("Frequency:  " + prt_str % prt_tpe)
            else:
                print(" " * 12 + prt_str % prt_tpe)
        print("=" * 80)
    return frequency

def extract(conn, table, column, size, shuffle=True, **query):
    '''extracts distil_id of desired data which restricted by data size and specified query condition.
    the first item of <query> determined the selected tags for classification.
    <shuffle> means whether shuffle the order of retrieved rows of data or not.
    '''
    # generates desired query commmand
    cmd_first = "SELECT distil_id, {0} FROM {1}".format(column, table)
    cmd_last = query_str(query)
    cmd = cmd_first + cmd_last

    print "QUERY_CMD: {0}".format(cmd)
    cur = conn.cursor()
    cur.execute(cmd)
    result = [(str(x[0]), str(x[1])) for x in cur.fetchall()]
    if shuffle:
        import random
        random.shuffle(result)
    distil, tags = zip(*result)

    scope = query[column]
    limit = size / len(scope)

    distil = np.array(list(distil))
    tags = np.array(list(tags))

    cid, label, linfo = [], [], [] # linfo means the acturally name of labels
    for i,each in enumerate(scope):
        tmp_inds = np.where(tags == each.strip('"'))[0][:limit]
        cid.extend(list(distil[tmp_inds]))
        label.extend([i] * len(tmp_inds))
        linfo.extend([each] * len(tmp_inds))
    label = np.array(label)
    linfo = np.array(linfo)
    return cid, label, linfo
