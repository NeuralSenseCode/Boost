

from matplotlib import font_manager
import matplotlib.ticker as tck
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from nltk.stem import WordNetLemmatizer
from clean import *

#Venera900 = 'Font/Venera/Venera 900/Venera-900.otf'  
#Venera100 = 'Font/Venera/Venera 100/Venera-100.otf'
#font_manager.fontManager.addfont(Venera900)
#font_manager.fontManager.addfont(Venera100)
#font1 = font_manager.FontProperties(fname=Venera100)
#font2 = font_manager.FontProperties(fname=Venera900)

def time_proportions(input_folder,results_folder, tags = ['',]):
    #header("> Running: Plotting Proportions Time Series")
    out_path = f"{results_folder}"
    os.makedirs(out_path, exist_ok=True) 

    dfs = {f"{f[12:-4]}": pd.read_csv(f'{input_folder}{f}', low_memory = False) for f in os.listdir(f'{input_folder}') if '.csv' in f and all(x in f for x in tags)}

    for ad,df in dfs.items():
        print(f">> Now Plotting: {ad}")
        y_dim = 3.6417323
        x_dim = 13.3346457
        labels = ['Alpha','Eng','WL']

        colors = {'Alpha':'black',
             'Eng':'steelblue',
             'WL':'deepskyblue'}

        xs = {'Alpha':df['Time Alpha'].dropna().to_numpy(),
             'Eng':df['Time Engagement'].dropna().to_numpy(),
             'WL':df['Time Workload'].dropna().to_numpy()}
        ys = {'Alpha':df['Frontal Asymmetry Alpha Filtered'].dropna().to_numpy(),
             'Eng':df['High Engagement Proportion Filtered'].dropna().to_numpy(),
             'WL':df['Optimal Workload Proportion Filtered'].dropna().to_numpy()}

        plt.rcParams['axes.edgecolor']='#333F4B'
        plt.rcParams['axes.linewidth']=0.8
        plt.rcParams['xtick.color']='#333F4B'
        plt.rcParams['ytick.color']='#333F4B'
        plt.rcParams['text.color']='#333F4B'

        fig = plt.figure()

        ax = fig.add_subplot(111)

        # change the style of the axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_position(('outward', 8))
        ax.spines['bottom'].set_position(('outward', 5))

        axes = plt.gca()
        axes.get_xaxis().set_visible(False)

        #plt.title(title, fontsize=12)
        plt.xlim(0,xs['Alpha'].max())
        plt.ylim(0,100)
        plt.yticks(ha='left',rotation=90,fontsize=7,  )

        for label in labels:
            plt.plot(xs[label],ys[label],label = label, color = colors[label],linewidth=3)
        #if legend:
        #    plt.legend()

        fig.set_size_inches(x_dim, y_dim)
        plt.savefig(f'{out_path}plot_{ad}.png', dpi=300, transparent=True)
        plt.close()
    
    #header("> Completed: Plotting Proportions Time Series")


def bar(out_folder, xcol, y1col, bar1_color ='#dadfe2', xlabel = '', y1label = '', title = '', tags =['',], highlight = {'':'',}, y1lim =None, tag = None, footnote = None, footnoteLines=None, minorTicks = None):
    plt.rcParams['font.family'] = "Century Gothic"
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    width = 0.65

    x_data = xcol
    y1_data = y1col
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                  spines=['left','bottom'],
                  annotation_color = 'black',
                  x_labels=x_data,
                  ylabel=y1label,
                  title=title,
                  ylim=y1lim,
                  tickf = 12,
                  labelf = 12,
                  titlef = 12,
                  )

    bar1 = ax1.bar(x_data,y1_data, width= width, color= bar1_color, label = y1label)

    #plt.xticks(rotation=45, ha = 'right', color= 'black')

    if minorTicks is not None:
        ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())

    plt.savefig(f"{out_path}plot_bar_{title}{tag}.png", dpi=300, transparent=True, bbox_inches="tight" )
    
    plt.close()

    if footnote is not None:
        plt.figtext(0.5,0.5, footnote, ha="center", fontsize=8, bbox={"facecolor":'lightgrey', "alpha":0.5, "pad":5})

    plt.savefig(f"{out_path}plot_sig_{title}{tag}.png", dpi=300, transparent=True, bbox_inches="tight" )
    plt.close()
    print(f">>> Plotted: {title}")


def double_bar(out_folder, xcol, y1col, y2col, bar1_color ='#dadfe2', bar2_color='#dadfe2', xlabel = '', ylabel='', y1label = '', y2label = '', title = '', tags =['',], highlight = {'':'',}, y1lim =None, ymin =None ,ymax =None  ):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    #df = df.replace(to_replace=np.nan,value=0)
    #df = df.sort_values(by=y1col,ascending=True)
    width = 0.35

    #x_data = df[xcol].values
    #y1_data = df[y1col].values
    #y2_data = df[y2col].values

    x_data = xcol
    y1_data = y1col
    y2_data = y2col

    x = np.arange(len(x_data))  # the label locations
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                  spines=['left','bottom'],
                  annotation_color = 'black',
                  x_ticks=x,
                  x_labels=x_data,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  title=title,
                  ymin = ymin,
                  ymax = ymax,
                  )

    bar1 = ax1.bar(x- width/2,y1_data,width, color= bar1_color, label = y1label)
    bar2 = ax1.bar(x + width/2,y2_data, width, color= bar2_color, label = y2label)
    
    ax1.legend(loc=2, bbox_to_anchor=(0.01,1),fontsize=4)
    #ax2.legend(loc=2, bbox_to_anchor=(0.01,0.965),fontsize=4)

    plt.tight_layout()
    plt.savefig(f"{out_path}plot_group_bar_{title}.png", dpi=300, transparent=True )

    print(f">>> Plotted: {title}")


def triple_bar(out_folder, xcol, y1col, y2col, y3col, bar1_color ='#dadfe2', bar2_color='#dadfe2', bar3_color='#dadfe2', xlabel = '', ylabel='', y1label = '', y2label = '', y3label = '', title = '', tags =['',], highlight = {'':'',}, y1lim =None ):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    #df = df.replace(to_replace=np.nan,value=0)
    #df = df.sort_values(by=y1col,ascending=True)
    width = 0.25

    #x_data = df[xcol].values
    #y1_data = df[y1col].values
    #y2_data = df[y2col].values

    x_data = xcol
    y1_data = y1col
    y2_data = y2col
    y3_data = y3col

    x = np.arange(len(x_data))  # the label locations
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                  spines=['left','bottom'],
                  annotation_color = 'black',
                  x_ticks=x,
                  x_labels=x_data,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  title=title,
                  ylim=y1lim,
                  )

    bar1 = ax1.bar(x- width,y1_data,width, color= bar1_color, label = y1label)
    bar2 = ax1.bar(x,y2_data, width, color= bar2_color, label = y2label)
    bar3 = ax1.bar(x + width,y3_data, width, color= bar3_color, label = y3label)
    
    ax1.legend(loc=2, bbox_to_anchor=(0.01,1),fontsize=4)
    #ax2.legend(loc=2, bbox_to_anchor=(0.01,0.965),fontsize=4)

    plt.tight_layout()
    plt.savefig(f"{out_path}plot_group_bar_{title}.png", dpi=300, transparent=True )

    print(f">>> Plotted: {title}")

def group_bar(out_folder, df, xcol, y1col, y2col, bar1_color ='#dadfe2', bar2_color='#dadfe2', xlabel = '', y1label = '', y2label = '', title = '', tags =['',], highlight = {'':'',}, y1lim =None, y2lim= None ):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    df = df.replace(to_replace=np.nan,value=0)
    df = df.sort_values(by=y1col,ascending=False)
    width = 0.35

    x_data = df[xcol].values
    y1_data = df[y1col].values
    y2_data = df[y2col].values
    x = np.arange(len(x_data))  # the label locations
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                  spines=['left','right','bottom'],
                  x_ticks=x,
                  x_labels=x_data,
                  xlabel=xlabel,
                  ylabel=y1label,
                  title=title,
                  ylim=y1lim,
                  )

    ax2 = std_axes(ax1.twinx(),
                  ylabel=y2label,
                  ylim=y2lim,
                  )

    bar1 = ax1.bar(x- width/2,y1_data,width, color= bar1_color, label = y1label[:-3] )
    bar2 = ax2.bar(x + width/2,y2_data, width, color= bar2_color, label = y2label[:-3])
    
    ax1.legend(loc=2, bbox_to_anchor=(0,1),fontsize=4)
    ax2.legend(loc=2, bbox_to_anchor=(0,0.965),fontsize=4)
    
    plt.savefig(f"{out_path}plot_group_bar_{title}.png", dpi=300, transparent=True )

    print(f">>> Plotted: {title}")


def std_axes(ax, spines=[], annotation_color = 'white' , x_ticks=None, x_labels=None, xlabel=None, ylabel=None, title=None, ylim=None, ymin = None, ymax= None, titlef=12, tickf=6, labelf=8, second = False ):   
    # change the style of the axis spines
    plt.rcParams['font.family'] = "Century Gothic"
    #plt.rcParams['font.sans-serif'] = font1.get_name()
    #plt.rcParams['text.color'] = 'white'

    all = ['top','bottom','left','right']
    for spine in all:
        ax.spines[spine].set_linewidth(0.8)
        if spine in spines:
            #ax.spines[spine].set_position(('outward',0.5))
            ax.spines[spine].set_color(annotation_color)
        else:
            ax.spines[spine].set_visible(False)
   

    ax.set_xticks(x_ticks) if x_ticks is not None else None
    ax.set_xticklabels(x_labels, fontsize=tickf, rotation=45, ha = 'right', color= annotation_color) if x_labels is not None else None
    ax.tick_params(axis='both', labelsize=tickf, color= annotation_color, labelcolor= annotation_color)
    
    ax.set_xlabel(xlabel,fontsize=labelf, color=annotation_color) if xlabel is not None else None
    ax.set_ylabel(ylabel,fontsize=labelf, color=annotation_color, labelpad = 20) if ylabel is not None else None
    ax.set_ylabel(ylabel,fontsize=labelf, color=annotation_color, rotation=270, labelpad = 20) if ylabel is not None and second is True else None
    ax.set_title(title, fontsize=titlef, color=annotation_color) if title is not None else None 
    
    if ylim is not None:
        ax.set_ylim(0,ylim)
    elif ymin is not None:
        ax.set_ylim(ymin,ymax)  
    else:
        pass
    
    #ax.yaxis.grid()
    return ax


def str_extract_rows(df, col1, mark):
    for index, row in df.iterrows():
        df.at[index, col1] = row[col1][0:row[col1].find(mark)]
    return df


def pie(out_folder, vals, labels, colors, title=None, explode=None, startangle = 90, circle = 0.14,  annotation_color = 'white'):  
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)   

    print(f">> Running: Plotting Data for {title}")

    y_dim = 5 /2.54
    x_dim = 7.5 /2.54
    
    fig = plt.figure()

    ax = fig.add_subplot(111)
                
    ax.pie(vals,
           startangle = startangle,
           labels=labels,
           colors =colors,
           explode=explode,
           radius = 0.4*x_dim,
           autopct='%1.0f%%',
           pctdistance=0.83,
           labeldistance=1.1,
           wedgeprops=dict(width=circle*x_dim),
           textprops={'color':annotation_color,
                      'fontsize':14})
    
    ax.set_title(title, fontsize=14, pad=25, color = annotation_color) if title is not None else None 
    #ax.add_artist(centre_circle)

    plt.tight_layout()
    plt.savefig(f"{out_path}plot_pie_chart_{title}.png", dpi=300, transparent=True )

    print(f">>> Plotted: {title}")


def box(out_folder, vals, labels, colors,xlabel=None, ylabel=None, xlim=None,  title=None,  annotation_color = 'white', vline=None):  
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)   

    print(f">> Running: Plotting Data for {title}")
    
    fig = plt.figure()

    ax = std_axes(fig.add_subplot(111),
                  spines=['left','bottom'],
                  title=title
                  )
                
    bp = ax.boxplot(vals, patch_artist = True, vert = 0)
    ax.set_xlabel(xlabel,fontsize=8, color=annotation_color) if xlabel is not None else None   
    ax.set_xlim(xlim) if xlim is not None else None

    colors = [colors[l] for l in labels]

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
 
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color =annotation_color,
                    linewidth = 1.5,
                    linestyle =":")
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color =annotation_color,
                linewidth = 2)
 
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color =annotation_color,
                    linewidth = 3)
 
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                    color =annotation_color,
                    alpha = 0.5)
     
    # x-axis labels
    ax.set_yticklabels(labels)   
    ax.set_title(title, fontsize=14, pad=25, color = annotation_color) if title is not None else None 
    #ax.add_artist(centre_circle)
    plt.axvline(vline, linestyle='dashed', color=annotation_color,linewidth=1) if vline is not None else None  

    plt.tight_layout()
    plt.savefig(f"{out_path}plot_box_plot_{title}.png", dpi=300, transparent=True )

    print(f">>> Plotted: {title}")


def quadrant(out_folder, yy,yn,ny,nn, title=None,):  
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)   

    print(f">> Running: Plotting Quadrant for {title}")

    fig, ax = plt.subplots()

    size = 0.4
    vals = np.array([[yy,ny],[yn,nn]])
      
    # Set x-axis range
    ax.set_xlim((0,10))
    # Set y-axis range
    ax.set_ylim((0,10))
    # Draw lines to split quadrants
    ax.plot([5,5],[0,10], linewidth=4, color='slategray' )
    ax.plot([0,10],[5,5], linewidth=4, color='slategray' )
    ax.set_title(title, pad=50, fontsize = 18, color= 'white') if title is not None else None 
    #Title
    ax.text(5,11,'Reported', fontsize =14, fontweight = 'bold',ha='center', va='center',color = 'white')
    ax.text(-1,5,'Actual', fontsize =14, fontweight = 'bold',ha='center', va='center', rotation = 90, color = 'white')
    #Inner
    ax.text(2.5,10,'Yes', fontsize =12, fontweight = 'bold',ha='center', va='center', color = 'white')
    ax.text(7.5,10,'No', fontsize =12, fontweight = 'bold',ha='center', va='center', color = 'white')
    ax.text(0,2.5,'No', fontsize =12, fontweight = 'bold',ha='center', va='center',rotation = 90, color = 'white')
    ax.text(0,7.5,'Yes', fontsize =12, fontweight = 'bold',ha='center', va='center',rotation = 90, color = 'white')

    ax.text(2.5,7.5,f'{yy:1.0f}%', fontsize =30, fontweight = 'bold',ha='center', va='center', color = 'deepskyblue')
    ax.text(7.5,7.5,f'{yn:1.0f}%', fontsize =30, fontweight = 'bold',ha='center', va='center', color = 'deepskyblue')
    ax.text(2.5,2.5,f'{ny:1.0f}%', fontsize =30, fontweight = 'bold',ha='center', va='center',color = 'deepskyblue')
    ax.text(7.5,2.5,f'{nn:1.0f}%', fontsize =30, fontweight = 'bold',ha='center', va='center',color = 'deepskyblue')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{out_path}plot_quadrant_chart_{title}.png", dpi=300, transparent=True )
    
    print(f">>> Plotted: {title}")


def bar_line(out_folder, xcol, y1col, y2col, bar_color ='#dadfe2', line_color='#dadfe2', xlabel = '', y1label = '', y2label = '', title = '', tags =['',], highlight = {'':'',}, y1lim =None, y2min= None, y2max= None, tag = None ):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    width = 0.65

    x_data = xcol
    y1_data = y1col
    y2_data = y2col
    x = np.arange(len(x_data))  # the label locations
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                   annotation_color = 'black',
                  spines=['left','right','bottom'],
                  x_ticks=x,
                  x_labels=x_data,
                  xlabel=xlabel,
                  ylabel=y1label,
                  title=title,
                  ylim=y1lim,
                  tickf = 12,
                  labelf = 12,
                  titlef = 12,
                  )
    bar = ax1.bar(x,y1_data,width, color= bar_color, label = y1label)
    
    ax2 = std_axes(ax1.twinx(),
                   annotation_color = 'black',
                    ylabel=y2label,
                    ymin = y2min,
                    ymax = y2max,
                    tickf = 12,
                    labelf = 12,
                    titlef = 12,
                    second = True,
                    )

    #plt.axhline(y=0, color='steelblue', linestyle='--',)
    line = ax2.plot(x,y2_data, color= line_color, label = y2label)
    fsize = 6

    #ax1.legend(loc=2, bbox_to_anchor=(1.2,1),fontsize=fsize)
    #ax2.legend(loc=2, bbox_to_anchor=(1.2,(1-0.035/4*fsize)),fontsize=fsize)
    
    plt.tight_layout()
    #plt.savefig(f"{out_path}plot_bar_line_{title}.png", dpi=300)
    plt.savefig(f"{out_path}plot_bar_line_{title}{tag}.png", dpi=300, transparent=True )

    print(f">>> Plotted: {title}")


def bar_bar_line(out_folder, xcol, y1col, y2col, y3col, bar1_color ='#dadfe2', bar2_color ='#dadfe2', line_color='#dadfe2', xlabel = '', yax1label='', yax2label='', y1label = '', y2label = '', y3label = '', title = '', tags =['',], highlight = {'':'',}, y1lim =None, y3min= None, y2lim =None, y3max= None, tag = None ):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    width = 0.35

    x_data = xcol
    y1_data = y1col
    y2_data = y2col
    y3_data = y3col

    x = np.arange(len(x_data))  # the label locations
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                   annotation_color = 'black',
                  spines=['left','right','bottom'],
                  x_ticks=x,
                  x_labels=x_data,
                  xlabel=xlabel,
                  ylabel=yax1label,
                  title=title,
                  ylim=y1lim,
                  tickf = 12,
                  labelf = 12,
                  titlef = 12,
                  )

    bar1 = ax1.bar(x- width/2,y1_data,width, color= bar1_color, label = y1label)
    bar2 = ax1.bar(x + width/2,y2_data, width, color= bar2_color, label = y2label)


    ax2 = std_axes(ax1.twinx(),
                   annotation_color = 'black',
                    ylabel=yax2label,
                    ymin = y3min,
                    ymax = y3max,
                    tickf = 12,
                    labelf = 12,
                    titlef = 12,
                    second = True,
                    )

    line = ax2.plot(x,y3_data, color= line_color, label = y3label)
    fsize = 6

    ax1.legend(loc=2, bbox_to_anchor=(1.2,1),fontsize=fsize)
    ax2.legend(loc=2, bbox_to_anchor=(1.2,(1-2*0.035/4*fsize)),fontsize=fsize)
    
    plt.tight_layout()
    #plt.savefig(f"{out_path}plot_bar_line_{title}.png", dpi=300)
    plt.savefig(f"{out_path}plot_bar_line_{title}_{tag}.png", dpi=300, transparent=True )

    print(f">>> Plotted: {title}")

def bar_bar(out_folder, xcol, y1col, y2col, bar1_color ='#dadfe2', bar2_color ='#dadfe2', xlabel = '', yax1label='', yax2label='', y1label = '', y2label = '', title = '', tags =['',], highlight = {'':'',}, y1lim =None, y2lim =None, tag = None ):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    width = 0.35

    x_data = xcol
    y1_data = y1col
    y2_data = y2col

    x = np.arange(len(x_data))  # the label locations
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                   annotation_color = 'black',
                  spines=['left','right','bottom'],
                  x_ticks=x,
                  x_labels=x_data,
                  xlabel=xlabel,
                  ylabel=yax1label,
                  title=title,
                  ylim=y1lim,
                  tickf = 12,
                  labelf = 12,
                  titlef = 12,
                  )

    bar1 = ax1.bar(x- width/2,y1_data,width, color= bar1_color, label = y1label)
    

    ax2 = std_axes(ax1.twinx(),
                   annotation_color = 'black',
                    ylabel=yax2label,
                    tickf = 12,
                    labelf = 12,
                    titlef = 12,
                    second = True,
                    )

    bar2 = ax2.bar(x + width/2,y2_data, width, color= bar2_color, label = y2label)
    fsize = 6

    ax1.legend(loc=2, bbox_to_anchor=(1.2,1),fontsize=fsize)
    ax2.legend(loc=2, bbox_to_anchor=(1.2,(1-2*0.035/4*fsize)),fontsize=fsize)
    
    plt.tight_layout()
    #plt.savefig(f"{out_path}plot_bar_line_{title}.png", dpi=300)
    plt.savefig(f"{out_path}plot_bar_bar_{title}_{tag}.png", dpi=300, transparent=True )

    print(f">>> Plotted: {title}")


def multi_bar(out_folder, x_data, y_data, xlabel = '', ylabel='', title = '', tags =['',], highlight = {'':'',}, ylim =None, footnote = None, footnoteLines=None, tag = None, minorTicks=None):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")
    plt.rcParams.update({'font.size': 22})

    n = len(y_data)
    pad = 0.1
    width = (1-2*pad)/n
    x_placement = np.arange(-0.5+(pad+(width/2)),0.5, width)

    x = np.arange(len(x_data))  # the label locations
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                  spines=['left','bottom'],
                  annotation_color = 'black',
                  x_ticks=x,
                  x_labels=x_data,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  title=title,
                  ylim=ylim,
                  )

    i=0
    for y in y_data:
        x_adjusted = x+[x_placement[i]]
        ax1.bar(x_adjusted ,y_data[y]['values'],width, color= y_data[y]['bar_color'], label = y_data[y]['label'])
        i += 1
    
    ax1.legend(loc=2, bbox_to_anchor=(0.01,1),fontsize=10)

    if minorTicks is not None:
        ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())

    ax1.set_xticklabels(x_data,fontsize=12)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)

    if tag is not None:
        plt.savefig(f"{out_path}plot_multibar_{title}_{tag}.png", dpi=300, transparent=True,bbox_inches="tight" )
    else:
        plt.savefig(f"{out_path}plot_multibar_{title}.png", dpi=300, transparent=True,bbox_inches="tight" )

    plt.close()
    plt.clf()
    if footnote is not None:
        plt.figtext(0.5, 0.5, footnote, ha="center", fontsize=8, bbox={"facecolor":'lightgrey', "alpha":0.5, "pad":5})

    if tag is not None:
        plt.savefig(f"{out_path}plot_sig_{title}_{tag}.png", dpi=300, transparent=True,bbox_inches="tight" )
    else:
        plt.savefig(f"{out_path}plot_sig_{title}.png", dpi=300, transparent=True,bbox_inches="tight" )
    plt.close()
    plt.clf()
    print(f">>> Plotted: {title}")


def multi_box(out_folder, x_data, y_data, xlabel = '', ylabel='', title = '', tags =['',], highlight = {'':'',}, ylim =None, footnote = None, footnoteLines=None, tag = None, minorTicks=None):
    out_path = f"{out_folder}"
    os.makedirs(out_path, exist_ok=True)    
    
    print(f">> Running: Plotting Data for {title}")

    n = len(y_data)
    pad = 0.1
    width = (1-2*pad)/n
    x_placement = np.arange(-0.5+(pad+(width/2)),0.5, width)

    x = np.arange(len(x_data))  # the label locations
    xb = np.arange(len(x_data)+1)  # the label locations
    
    fig = plt.figure()
    ax1 = std_axes(fig.add_subplot(111),
                  spines=['left','bottom'],
                  annotation_color = 'black',
                  x_ticks=x,
                  x_labels=x_data,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  title=title,
                  ylim=ylim,
                  )

    i=0
    for y in y_data:
        x_adjusted = x+[x_placement[i]]
        bp = ax1.boxplot(y_data[y]['values'], patch_artist = True)#,width, color= y_data[y]['bar_color'], label = y_data[y]['label'])
        i += 1
       
        ax1.set_xticks(xb)
        ax1.set_xticklabels(['',]+x_data)
        ax1.set_xlim(0.5)

        #plt.show()
    
        colors = [y_data[y]['bar_color'],]*len(x_data)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
 
        # changing color and linewidth of
        # whiskers
        for whisker in bp['whiskers']:
            whisker.set(color ='black',
                        linewidth = 1.5,
                        linestyle =":")
 
        # changing color and linewidth of
        # caps
        for cap in bp['caps']:
            cap.set(color ='black',
                    linewidth = 2)
 
        # changing color and linewidth of
        # medians
        for median in bp['medians']:
            median.set(color ='black',
                        linewidth = 3)
 
        # changing style of fliers
        for flier in bp['fliers']:
            flier.set(marker ='D',
                        color ='black',
                        alpha = 0.5)

    #ax1.legend(loc=2, bbox_to_anchor=(0.01,1),fontsize=4)

    if minorTicks is not None:
        ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())

    if tag is not None:
        plt.savefig(f"{out_path}plot_box_{title}_{tag}.png", dpi=300, transparent=True,bbox_inches="tight" )
    else:
        plt.savefig(f"{out_path}plot_box_{title}.png", dpi=300, transparent=True,bbox_inches="tight" )

    plt.close()

    if footnote is not None:
        plt.figtext(0.5, 0.5, footnote, ha="center", fontsize=8, bbox={"facecolor":'lightgrey', "alpha":0.5, "pad":5})

    if tag is not None:
        plt.savefig(f"{out_path}plot_sig_{title}_{tag}.png", dpi=300, transparent=True,bbox_inches="tight" )
    else:
        plt.savefig(f"{out_path}plot_sig_{title}.png", dpi=300, transparent=True,bbox_inches="tight" )
    
    plt.close()

    print(f">>> Plotted: {title}")




if __name__ == "__main__":
    main()