import plotly.graph_objects as go
import plotly.express as px


def make_plot(data, x, y, title, x_label, y_label, kind='line', color=None):
    if kind == 'line':
        fig = px.line(data, x=x, y=y, height=600, width = 1000,markers=True, title=title)
    elif kind == 'scatter':
        fig = px.scatter(data, x=x, y=y, height=600, width = 1000, title=title)
    elif kind == 'bar':
        fig = px.bar(data, x=x, y=y, height=600, width = 1000, title=title)
    else:
        raise ValueError("Unsupported plot type. Use 'line', 'scatter', or 'bar'.")
    
    if color:
        fig.update_traces(marker_color=color)
    
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    fig.show()
    fig.write_html(f"charts/{title}.html")



def make_plot_2y(data, x, y, y2, title, x_label, y_label, kind='line', color=None):
    if kind == 'line':
        fig = px.line(data, x=x, y=y, height=600, width = 1000,markers=True, title=title)
    elif kind == 'scatter':
        fig = px.scatter(data, x=x, y=y, height=600, width = 1000, title=title)
    else:
        raise ValueError("Unsupported plot type. Use 'line', 'scatter'.")

    if fig:
        fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[y2],
                    mode='lines+markers',
                    name=y2,
                    )
        )
        # Update layout to include multiple y-axes
        fig.update_layout(
            yaxis2=dict(
                title='Y-Axis 2',
                overlaying='y',
                side='right'
                )
        )
        
        if color:
            fig.update_traces(marker_color=color)
        
        fig.show()

        #Save file in the charts folder in html format  
        fig.write_html(f"charts/{title}.html")
