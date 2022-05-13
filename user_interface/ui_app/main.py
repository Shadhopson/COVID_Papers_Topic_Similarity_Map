#Imports for loading data into the app. Housed in the data.py file in the same directory as this main.py

# Imports for setting up the configuration screen. Housed in the config_menu.py file

# General imports
import json
import numpy as np
import pandas as pd
import math
import random
from itertools import chain, combinations
import networkx
import pandas_bokeh
import re as re

from bokeh.io import output_file, show, curdoc
from bokeh.models import ColumnDataSource,CDSView, FactorRange, GraphRenderer, Range1d, Circle, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet, Select, BooleanFilter, Button, GroupFilter, Div, RadioButtonGroup, CustomJS, IndexFilter
from bokeh.plotting import figure, show
from bokeh.models.tools import BoxZoomTool, ResetTool, BoxSelectTool, HoverTool, MultiLine, Range1d, TapTool
from bokeh.layouts import row, column, gridplot, layout
from bokeh.models.graphs import StaticLayoutProvider
from bokeh.models.widgets import Tabs, Panel, DataTable, TableColumn, HTMLTemplateFormatter
from bokeh.palettes import Spectral4, Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.plotting import from_networkx, output_file, save
from bokeh.plotting import from_networkx
from bokeh.transform import linear_cmap
from networkx.algorithms import community
from functools import partial
from sqlalchemy import create_engine
import pymysql
# Any general imports
from functools import partial # Needed to allow for the passing of additional arguments to widget callbacks

#######################################################################################################
#paper metadata for citation table #SQL

def load_sample_paper_data():
    # paper_df_path = "d:/users/AThomas/Documents/cse6242_team094_dnd-viz-prototype/ui/sample_data/metadata_sample_5000.csv"
    # paper_df = pd.read_csv(paper_df_path, sep=",")
    paper_df = pd.read_sql_table('metadata', con=dbConnection)
    abstract_df = pd.read_sql_table('processed_abstracts', con=dbConnection)
    titles_df = pd.read_sql_table('titles', con=dbConnection)

    # paper_data_df = paper_df.join(abstract_df, on="cord_uid")
    # combine both of these and drop dupe columns
    paper_data_df = pd.concat([paper_df, abstract_df, titles_df], axis=1)
    paper_data_df = paper_data_df.loc[:,~paper_data_df.columns.duplicated()]

    return paper_data_df

###################################from config_menu##########################################

##datasources##change back
BERT = '02a_bert_string_doc_to_topic', '02a_bert_string_topic_to_words'
LDA = '02b_lda_string_doc_to_topic', '02b_lda_string_topic_to_words'

# Configure MySQL Connection change back
sqlEngine = create_engine('mysql+pymysql://root:p@ssw0rd1@cse6242_team094_mysqldb/cse6242_team094')
dbConnection = sqlEngine.connect()

##Topic model info
bert_node_info = pd.read_sql_table('03_bert_topic_clust_output', con=dbConnection)
bert_node_pagerank = pd.read_sql_table('03_bert_topic_pagerank', con=dbConnection)
bert_edgelist = pd.read_sql_table('03_bert_topic_graph', con=dbConnection)



lda_node_info = pd.read_sql_table('03_lda_topic_clust_output', con=dbConnection)
lda_node_pagerank = pd.read_sql_table('03_lda_topic_pagerank', con=dbConnection)
lda_edgelist = pd.read_sql_table('03_lda_topic_graph', con=dbConnection)


##merge node info
bert_node_list = pd.merge(bert_node_info, bert_node_pagerank, how="inner", on="node_id")##add in paper info, which means find top papers, then find paper info
lda_node_list = pd.merge(lda_node_info, lda_node_pagerank, how="inner", on="node_id")



source1b, source2b = BERT 




def show_sample_data_table(source_data, source_view):
    """
    Use this function to display a table of the original data produced by the graph algorithm step.
    """
    template = """<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""
    columns = [
            TableColumn(field="title", title="Title", formatter=HTMLTemplateFormatter(template=template)),
            TableColumn(field="authors", title="Authors", formatter=HTMLTemplateFormatter(template=template)),
            TableColumn(field="publish_time", title="Date", formatter=HTMLTemplateFormatter(template=template)),
            TableColumn(field="journal", title="Journal", formatter=HTMLTemplateFormatter(template=template)),
            TableColumn(field="abstract", title="Abstract", formatter=HTMLTemplateFormatter(template=template)),
            TableColumn(field="url", title="URL", formatter=HTMLTemplateFormatter(template=template)),
            TableColumn(field="cord_uid", title="Paper_id"),
            TableColumn(field = 'Topic', title = 'Topic')
        ]
    data_table = DataTable(source=source_data, view=source_view, columns=columns, width = 800, row_height = 60)

    return data_table




###filter for graph views
topic_filter = '1'


def make_data_sources(NLP_Type):
    source1, source2 = NLP_Type
#read in data
    df = pd.read_sql_table(source1, con=dbConnection)
    if 'abstract' in df:
        del df['abstract']
    df.columns=['paper_count','cord_uid','topic','topic_prob']
    df2 = df#can add slices to debug
    topic_paper = df.filter(['topic', 'cord_uid'])
    topic_paper.columns = ['Topic', 'cord_uid']
#making the groups and labels for the graph
    papers = df2['cord_uid']

#account for topic of -1 as label of topics
    topics_pd = df['topic']
    topics = list(set([topic for topic in topics_pd]))
    topics.sort()
#extracting probabilities list of lists. so ugly. how do i make topic_prob easier to use?
    prob = df2['topic_prob']

    probs = []
    for each in prob:
        each.strip('\n')
        s = each.replace("[","").replace("]", "")
        thing = s.split()
        probs.append(thing)
#create k, v pairs of topic name to probabilities, sort by values and create list of dicts
    listdict = []
    testdict = []
    for prob in probs:
        testdict1 = dict(zip(topics, prob))
        testdict.append(testdict1)
        sorted_tuples = sorted(testdict1.items(), key = lambda item: float(item[1])*-1)
        sorted_dict = {k:v for k, v in sorted_tuples[:5]} #the index limits to top n **HOW DOES THIS NEED TO CHANGE FOR UI
        listdict.append(sorted_dict)

#assign each paper to its labeled, sorted list of probabilities
    bigdict = dict(zip(papers, listdict))
#End top topics by paper dict################################################################################################
##############################################################################################################################

#Begin top papers by topic###################################################################################################
    top2pap_dict = dict(zip(papers, testdict))

#create df of probabilities
    topic_split_df = pd.DataFrame.from_dict(top2pap_dict)

#find top n papers per topic ******************n should be changeable for UI
    n=5
    Tops = topic_split_df.apply(lambda x:list(topic_split_df.columns[np.array(x).argsort()[::-1][:n]]), axis=1).to_list()


###combine top papers labels with paper info
    top2pap2tops=[]
    for each in Tops:
        newdict ={}
        for paper in each:
            newdict[paper] = bigdict[paper]
        top2pap2tops.append(newdict)

    t2p2t_dict = dict(zip(topics, top2pap2tops))

#convert to Df source https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    top2pap2tops_DF = pd.DataFrame.from_records(
        [
            (level1, level2, level3, leaf)
            for level1, level2_dict in t2p2t_dict.items()
            for level2, level3_dict in level2_dict.items()
            for level3, leaf in level3_dict.items()
        ],
        columns=['Topic', 'Paper', 'Top_by_paper', 'prob_top_ass2pap']
    )

#convert prob to float
    top2pap2tops_DF['prob_top_ass2pap'] = top2pap2tops_DF['prob_top_ass2pap'].astype(float)
###remove very low probabilities
    filt_top2pap2tops_DF= top2pap2tops_DF[top2pap2tops_DF['prob_top_ass2pap']>0.000001]
############################################################################################################################################################
############################################################################################################################################################
###from tina's# load data

# load data
    topic_df = pd.read_sql_table(source2, con=dbConnection)
    if 'Count' in topic_df:
        del topic_df['Count']
    topic_df.columns=['topic_count','Topic','related_words']
# use first row for now
    topic_df_test = topic_df
# turn columns we need for now into series
    related_words_series = pd.Series(topic_df_test['related_words'])
    topic_series = pd.Series(topic_df_test['Topic'])

# turn these columns into a dataframe
    frame = { 'Topic': topic_series, 'Related_Words': related_words_series}
    result = pd.DataFrame(frame)


# clean data by removing special characters
    spec_chars =['"',"[","]","(",")"]
    spec_chars = ['"',"[","]"] #keeping parenth

    for char in spec_chars:
        result['Related_Words'] = result['Related_Words'].str.replace(char, '', regex=False)

# 
# ########### place probabilities and topic words in separate columns ##################

# function to extract only integers of the column
    def find_prob(text):
        num = re.findall(r'[0.0-9]+', text)
        return " ".join(num)

# apply function to our column and save into new column named topic_prob
    result['topic_prob'] = result['Related_Words'].apply(lambda x: find_prob(x))
# stack probabilities by topic to make it easier
    out2 = (result.set_index('Topic')['topic_prob']
                .str.split(' ', expand=True)
                .stack()
                .rename('Topic')
                .reset_index(name='topic_prob'))

# convert topic_prob column to float
#changing name for future merge
    out2['word_prob'] = out2['topic_prob'].astype(float)
    out2['word_prob'] = out2['word_prob'] ###***
    del out2['topic_prob']
# extract only the words from the column now
    def find_words(text):
        results = re.findall(r'\b[^\d\W]+\b', text)
        return " ".join(results)

# apply function to our column and save into new column named words
    result['words'] = result['Related_Words'].apply(lambda x: find_words(x))
    related_words_df = result['words']

# stack results
    out3 = (result.set_index('Topic')['words']
                .str.split(' ', expand=True)
                .stack()
                .rename('Topic')
                .reset_index(name='words'))



    final_dat = pd.concat([out2, out3], axis=1)
    final_dat = final_dat.drop('level_1',1)
    final_dat = final_dat.loc[:,~final_dat.columns.duplicated()]


    topic_word_dict = final_dat.groupby('Topic').apply(lambda x: x.to_dict('r')).to_dict()


###############################################################merged dataframe############################
    df_outer = pd.merge(final_dat, filt_top2pap2tops_DF,on='Topic', how='outer')
    df_outer['Topic'] = df_outer["Topic"].astype(str)

    
    return t2p2t_dict, topic_word_dict, bigdict, related_words_df, top2pap2tops_DF, topic_paper, topics

##*************************************************************************this

##TOPIC LABEL -1 IS A REAL PROBLEM DOUBLE CHECK EVERYTHING!!!!!!

#######################procesing bert and lda#############################################################
bert_t2p2t_dict, bert_topic_word_dict, bert_bigdict, bert_related_words_df, bert_top2pap2tops_DF, bert_topic_paper, bert_topics  = make_data_sources(BERT)


lda_t2p2t_dict, lda_topic_word_dict, lda_bigdict, lda_related_words_df, lda_top2pap2tops_DF, lda_topic_paper, lda_topics = make_data_sources(LDA)

#########################START OF REVISED GRAPHS###############################################################################


def paper_topic_chart(filter, t2p2t_dict):
    pairings=[]
    flatprob = []
    for k, v in t2p2t_dict.items():
        if str(k) == filter:
            
            for paper, topics in v.items():
                name = paper
                for key, values in topics.items():
                    tup = name, str(key)
                    pairings.append(tup)
                    flatprob.append(values)
    return pairings, flatprob

bert_pairings, bert_flatprob = paper_topic_chart(topic_filter, bert_t2p2t_dict)
lda_pairings, lda_flatprob = paper_topic_chart(topic_filter, lda_t2p2t_dict)

        
 ###########################bar chart topics by paper##############################       
def paper_to_topic_plot(pairings, flatprob):
    source = ColumnDataSource(data = dict(x=pairings, counts = flatprob))

    view1 = CDSView(source = source, filters = []) ###add in words when hover over topic


    hov_info = [('topic', '@x'), ('probability', '@counts')]


    p = figure(x_range=FactorRange(*pairings), 
            plot_height=300, width = 800, title="Top papers by selected topic with each paper's most relevant topics",tooltips = hov_info,
            tools=[BoxZoomTool(), 'save,reset'])

    p.vbar(x='x', top='counts', width=0.9, source=source, view=view1)


    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Paper ID with Top Topics by Paper"
    p.title.text_font_size = '16pt'

    return p
###make plot for bert and lda
bert_p = paper_to_topic_plot(bert_pairings, bert_flatprob)
lda_p = paper_to_topic_plot(lda_pairings, lda_flatprob)

####bar charts for words by topics ################################# make a barchart for each topic

def words_chart(filter, topic_word_dict):
    pairings2 = []
    flatprob2 = []
    for topic in topic_word_dict.items():
        if str(topic[0]) == filter:
            name = str(f'Topic {topic[0]}')
            for each in topic[1]:
                tup = name, each['words']
                pairings2.append(tup)
                flatprob2.append(each['word_prob'])
    return pairings2, flatprob2

bert_pairings2, bert_flatprob2 = words_chart(topic_filter, bert_topic_word_dict)
lda_pairings2, lda_flatprob2 = words_chart(topic_filter, lda_topic_word_dict)

##################making word freq bar chart############
#### keeping this for now since I dont have the code for LDA bar chart right now
def word_bars(pairings2, flatprob2):
    source2 = ColumnDataSource(data = dict(info= pairings2, prob = flatprob2))
    view2 = CDSView(source = source2, filters = [GroupFilter(column_name = "Topic", group = '1')])

    hov_info2 = [('topic', '@info'), ('in-topic frequency', '@prob')]
    b = figure(x_range=FactorRange(*pairings2), 
            #sizing_mode = "stretch_width", #allows automatic stretching to browser width
            plot_height=300, width=600, title="Top words by selected topics",tooltips = hov_info2, toolbar_location= 'right',
            tools=[BoxZoomTool(), ResetTool(), 'pan'])

    b.vbar(x='info', top='prob', width=0.9, source=source2, view = view2)

    b.y_range.start = 0
    b.x_range.range_padding = 0.1
    b.xaxis.major_label_orientation = 1
    b.xgrid.grid_line_color = None
    b.xaxis.axis_label = "Top Words by Selected Topic"
    b.title.text_font_size = '16pt'
    return b

lda_b = word_bars(lda_pairings2, lda_flatprob2)


############################################## UPDATED bar chart #################################
# only BERT for now
# I need final_dat for this bar chart
# load data

topic_df = pd.read_sql_table(source2b, con=dbConnection)

# turn columns we need for now into series
related_words_series = pd.Series(topic_df['related_words'])
topic_series = pd.Series(topic_df['Topic'])

# turn these columns into a dataframe
frame = { 'Topic': topic_series, 'Related_Words': related_words_series}
result = pd.DataFrame(frame)

# clean data by removing special characters
spec_chars =['"',"[","]","(",")"]

for char in spec_chars:
    result['Related_Words'] = result['Related_Words'].str.replace(char, '', regex=False)



########### place probabilities and topic words in separate columns ##################

# function to extract only integers of the column
def find_prob(text):
    num = re.findall(r'[0.0-9]+', text)
    return " ".join(num)

# apply function to our column and save into new column named topic_prob
result['topic_prob'] = result['Related_Words'].apply(lambda x: find_prob(x))

# stack probabilities by topic to make it easier
out2 = (result.set_index('Topic')['topic_prob']
              .str.split(' ', expand=True)
              .stack()
              .rename('Topic')
              .reset_index(name='topic_prob'))

# convert topic_prob column to float
out2['topic_prob'] = out2['topic_prob'].astype(float)


# extract only the words from the column now
def find_words(text):
    results = re.findall(r'\b[^\d\W]+\b', text)
    return " ".join(results)

# apply function to our column and save into new column named words
result['words'] = result['Related_Words'].apply(lambda x: find_words(x))

# stack results
out3 = (result.set_index('Topic')['words']
              .str.split(' ', expand=True)
              .stack()
              .rename('Topic')
              .reset_index(name='words'))


# combine both of these and drop dupe columns
final_dat = pd.concat([out2, out3], axis=1)
final_dat = final_dat.loc[:,~final_dat.columns.duplicated()]

# delete level_1 column
del final_dat['level_1']

# only have BERT for now...

# add the word Topic to all topics in the column
final_dat['Topic'] = 'Topic ' + final_dat['Topic'].astype(str)


# Need to create some type of function that will loop through the topics...
filter = final_dat['Topic'].unique().tolist() # how do we make this flexible?

# BERT
# take columns from our final_dat DF for source
words_list = final_dat.loc[final_dat['Topic'] == filter[0], 'words'].iloc[:]
prob_list = final_dat.loc[final_dat['Topic'] == filter[0], 'topic_prob'].iloc[:]
source = ColumnDataSource(data = dict(words=words_list, prob=prob_list))





# set plot
hov_info2 = [('Word', '@words'), ('Probability', '@prob')]
bert_b = figure(x_range=words_list, plot_height=300, width=600, title="Top 10 Words by Selected Topic", tooltips=hov_info2,
toolbar_location='right', tools=[BoxZoomTool(), ResetTool(), 'pan'])
v = bert_b.vbar(x='words', top='prob', width=0.6, source=source)
bert_b.xaxis.axis_label = "Top Words by Selected Topic"
bert_b.title.text_font_size = '16pt'


# Makes a unique list of Topics
options = []
options.extend(final_dat['Topic'].unique().tolist())

# function to update the plots. Why u no work?
def update_plot(attr, old, new):
    if select.value=='Topic 1':
        df_filter = final_dat.copy()
    else:
        df_filter = final_dat[final_dat['Topic']==select.value]

    words_list2 = df_filter.loc[df_filter['Topic'] == filter[0], 'words'].iloc[:]
    prob_list2 = df_filter.loc[df_filter['Topic'] == filter[0], 'topic_prob'].iloc[:]

    source1 = ColumnDataSource(data=dict(words=words_list2, prob=prob_list2))
    v.data_source.data = source1.data


# select widget
select = Select(title="Select a Topic", options=options, value="Topic")
select.on_change('value', update_plot)
controls = select



##############################END WORD GRAPH########################################################################################



##################################################################################network graphs from scotts######################################
def make_network_graph(node_list, edgelist, related_words_df):    

    spec_chars =['"',"[","]","(",")"]

    cluster_filter_value = 12
    cluster_filter = node_list['num_clusters'] == cluster_filter_value

    node_list_filt = node_list[cluster_filter].reset_index()
    node_list_filt = node_list_filt.astype({'node_id':str})

########### place probabilities and topic words in separate columns ##################

# function to extract only integers of the column
    def find_top(text):
        num = re.findall(r'[0.0-9]+', str(text))
        return " ".join(num)
# apply function to our column and save into new column named topic_prob
    for i in range(len(edgelist)):
        edgelist.iloc[i]= edgelist.iloc[i].apply(lambda x: find_top(x))

######Clustered Network Graph layout/creation######################################################################################################
##Data
    C = networkx.from_pandas_edgelist(edgelist, 'source', 'target', 'weight')

    degrees = dict(networkx.degree(C))
    networkx.set_node_attributes(C, name='degree', values=degrees)

###The following adjusts node size UP so small nodes visible. Will leave out for now and see how works with zoom
    number_to_adjust_by = 5
    adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in networkx.degree(C)])
    networkx.set_node_attributes(C, name='adjusted_node_size', values=adjusted_node_size)
##add clus id to node attributes
    CI = []
    for n, x in enumerate(node_list_filt['cluster_id']):
        pair = str(n), x
        CI.append(pair)
    adjusted_clus_info = dict(CI)
    networkx.set_node_attributes(C, name='cluster_id', values=adjusted_clus_info)

##add pagerank to node attributes
    pagerank = []
    for n, x in enumerate(node_list_filt['page_rank_val']):
        pair = str(n), format(x, 'f')
        pagerank.append(pair)
    adjusted_PR_info = dict(pagerank)
    networkx.set_node_attributes(C, name='page_rank', values=adjusted_PR_info)

################setting color and size info######################################################################

#Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8, Viridis256, GnBu, Viridis8
    color_palette = Blues8
#

#Choose colors for node and edge highlighting
    node_highlight_color = 'white'
    edge_highlight_color = 'black'

#Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
    PR = []
    if len(node_list) < 30:
        for n, x in enumerate(node_list_filt['page_rank_val']):
            pair = str(n), x/100
            PR.append(pair)
        PR_size = dict(PR)
        networkx.set_node_attributes(C, name='PR_size', values=PR_size)
    else:
        for n, x in enumerate(node_list_filt['page_rank_val']):
            pair = str(n), x*1000#**1.2
            PR.append(pair)
        PR_size = dict(PR)
        networkx.set_node_attributes(C, name='PR_size', values=PR_size)


###add words to node info
    RW = []
    for n, x in enumerate(list(related_words_df)):
        pair = str(n-1), x
        RW.append(pair)
    RW_dict = dict(RW)
    networkx.set_node_attributes(C, name='related_words', values=RW_dict)

    
    size_by_this_attribute= 'PR_size'
   
    color_by_this_attribute = 'cluster_id'
############linking data to figure####################################################################################


    title = "Covid-19 Topic Relationships"
    HOVERTOOLTIPS = [('Topic', '@index'), ('Related Words','@related_words'), ('Degree', '@degree'), ('Page Rank', '@page_rank')]#

    plot2 = figure(tooltips = HOVERTOOLTIPS, tools = 'pan,wheel_zoom,save,reset', active_scroll = 'wheel_zoom',
                    x_range = Range1d(-10.1, 10.1), y_range = Range1d(-10.1, 10.1), title = title, toolbar_location= 'right', height = 600, width = 600)

#Create a network graph object with spring layout
# https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
    network_graph = from_networkx(C, networkx.spring_layout, scale=10, center=(0, 0))

#Set node size and color by degree
#Set node sizes and colors according to node degree and pagerank (color as spectrum of color palette)
    minimum_value_color = min(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[color_by_this_attribute])

#flipped min and max value color so bigger would be darker
    network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=linear_cmap(color_by_this_attribute, color_palette, minimum_value_color, maximum_value_color))

#for coloring by modularity
#network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color= color_by_this_attribute)

#Set node highlight colors
    network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
    network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)


#Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

#Set edge highlight colors
    network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
    network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)

#Highlight nodes and edges
    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()

#Add Labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(C.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x='x', y='y', text='name', source=source, text_font_size='10px', background_fill_alpha=.7) #, background_fill_color='white'
    plot2.renderers.append(labels)

    plot2.title.text_font_size = '16pt'

#Add network graph to the plot
    plot2.renderers.append(network_graph)

    return plot2

##create network graphs for bert and lda
bert_plot2 = make_network_graph(bert_node_list, bert_edgelist, bert_related_words_df)
lda_plot2 = make_network_graph(lda_node_list, lda_edgelist, lda_related_words_df)


###############################end bert topic clusters graph######################################


###########################################radio group button widget between bert and lda######################################
#https://stackoverflow.com/questions/53565530/changing-gridplot-using-radiobuttongroup-python-bokeh

####fix whack formatting**
div = Div(text="""<h2><b><u>Natural Language Processing Methods:</u></h2>
<b>The tabs at the top of the page let you select between the two natural language processing methods (NLP) that were used to analyze the abstracts. 
NLP is a branch of AI that aims to use computer algorithms to process and decipher human languages. <br />The graphs on each tab are 
generated using the results from two of the current state-of-the-art NLP methods, BERT and LDA.<br /><br />
<b>BERT</b> is a natural language processing program that can analyze words in context to help categorize text.<br /><br />
<b>LDA</b> is a topic modeling algorithm that uses groups of words to map text to topics.<br />
""",style={'font-size': '110%'},
width=800, height=100)

h_div = Div(text = """<h1>COVID-19 Document Similarity</h1>""", width=800, height=100)
table_div = Div(text = """Note: You can view the full contents of each cell by moving your cursor over the cell of interest.""", width=500, height=100)

top_explore_div = Div(text="""<br /><h2><b><u>Further Topic Exploration:</u></b></h2>
Use the charts below to further investigate each topic. The first chart displays the top ten words by the selected topic. The second chart
displays the papers most closely aligned with each the selected topic. The chart also displays that paper's most closely associated topics 
so that you may discover topics associated through related papers. """,style={'font-size': '125%'}, width=800, height=100)


###############from orig main##########################################################
clustering_df = load_sample_paper_data()

#link paper citation info to related topic info
def create_table(top2pap2tops_DF, clustering_df, topic_paper):  
    one = top2pap2tops_DF.filter(['Topic', 'Paper'], axis=1)
    one = one.drop_duplicates(['Topic', 'Paper'])
    two = top2pap2tops_DF.filter(['Top_by_paper', 'Paper'], axis=1)
    two.columns = ['Topic', 'Paper']
    df = one.merge(two, on=['Topic', 'Paper'], how='outer')
    df = df.drop_duplicates(['Topic', 'Paper'])
    df.columns = ['Topic', 'cord_uid']
    df2 = df.merge(topic_paper, on=['Topic', 'cord_uid'], how='outer')
    df2 = df2.drop_duplicates(['Topic', 'cord_uid'])
    clustering_df['cord_uid'] = clustering_df['cord_uid']
    clustering_df2 = clustering_df.merge(df2, how= "outer", on = ["cord_uid"])
    clustering_df2a = clustering_df2.replace(np.nan, 100, regex = True)
    clustering_df2a['Topic'] = clustering_df2a['Topic'].astype(int)
    clustering_df2a['Topic'] = clustering_df2a['Topic'].astype(str)


    clustering_data_source = ColumnDataSource(clustering_df2a)
    clustering_data_view=CDSView(source=clustering_data_source, filters=[])
    ###make table w data
    paper_table = show_sample_data_table(clustering_data_source, clustering_data_view)
    ##my widget

    options = []
    options = sorted(clustering_df2a.Topic.unique())

    table_topic_select = Select(title = 'Select Topic', value = "", options = [""]+ options, syncable = True)

    def select_call2():#attr, old, new
        table_filter= table_topic_select.value
        update_top_filt = [GroupFilter(column_name='Topic', group =str(table_filter))]
        clustering_data_view.filters = update_top_filt


    button_label = "Update Table"
    update_button1 = Button(label=button_label, button_type="primary")
    update_button1.on_click(select_call2)

    return paper_table, table_topic_select, update_button1

l_paper_table, l_table_topic_select, l_update_button1 = create_table(lda_top2pap2tops_DF, clustering_df, lda_topic_paper)
b_paper_table, b_table_topic_select, b_update_button1 = create_table(bert_top2pap2tops_DF, clustering_df, bert_topic_paper)




#################################layout######################################################3

tab1 = Panel(child = column(h_div, div, row(column(b_table_topic_select,
        b_update_button1, b_paper_table, table_div), bert_plot2), top_explore_div, controls, row(bert_b, bert_p)), title = "BERT") #algo_widget,
tab2 = Panel(child = column(h_div, div, row(column(l_table_topic_select, l_update_button1, l_paper_table, table_div),lda_plot2), top_explore_div,controls, row(lda_b, lda_p)), title = "LDA")

divTemplate = Div(text="""
            <style>
            .bk.sent-later {
                font-size: 40px;
            }
            </style>
    """)
    


tabs = Tabs(tabs=[tab1, tab2])
output_file('Almost_done.html')

layout = layout([tabs])  ##in gridview for html can have merge_tools = False so each plot has own toolbar

curdoc().add_root(layout)


####################################################################################################################
