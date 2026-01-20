"""
Visualization Module
Creates publication-quality visualizations for Aadhaar analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class AadhaarVisualizer:
    """Create visualizations for Aadhaar data analysis"""
    
    def __init__(self, output_dir='../outputs/figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_time_series(self, df, value_col, title="Time Series", group_col='state', 
                        anomaly_col=None, save_name=None):
        """
        Plot time series with optional anomaly markers
        
        Args:
            df: DataFrame with time series data
            value_col: Column to plot
            title: Plot title
            group_col: Column to group by
            anomaly_col: Column with anomaly flags
            save_name: Filename to save plot
        """
        fig = go.Figure()
        
        # Plot each group
        for name, group in df.groupby(group_col):
            fig.add_trace(go.Scatter(
                x=group['date'],
                y=group[value_col],
                mode='lines',
                name=name,
                line=dict(width=2)
            ))
            
            # Add anomaly markers
            if anomaly_col and anomaly_col in group.columns:
                anomalies = group[group[anomaly_col] == 1]
                if len(anomalies) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomalies['date'],
                        y=anomalies[value_col],
                        mode='markers',
                        name=f'{name} Anomalies',
                        marker=dict(size=10, color='red', symbol='x')
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=value_col.replace('_', ' ').title(),
            hovermode='x unified',
            height=500
        )
        
        if save_name:
            fig.write_html(f"{self.output_dir}/{save_name}.html")
            fig.write_image(f"{self.output_dir}/{save_name}.png")
        
        return fig
    
    def plot_heatmap(self, df, value_col, title="Heatmap", save_name=None):
        """
        Create heatmap for state x time patterns
        
        Args:
            df: DataFrame with data
            value_col: Column to visualize
            title: Plot title
            save_name: Filename to save plot
        """
        # Pivot data
        pivot_df = df.pivot_table(
            values=value_col,
            index='state',
            columns=pd.to_datetime(df['date']).dt.to_period('M'),
            aggfunc='sum'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(pivot_df, cmap='YlOrRd', annot=False, fmt='.0f', 
                   cbar_kws={'label': value_col.replace('_', ' ').title()}, ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_choropleth_map(self, df, value_col, title="India Map", save_name=None):
        """
        Create choropleth map of India
        
        Args:
            df: DataFrame with state-level data
            value_col: Column to visualize
            title: Plot title
            save_name: Filename to save plot
        """
        # Aggregate by state
        state_data = df.groupby('state')[value_col].mean().reset_index()
        
        # Create choropleth
        fig = px.choropleth(
            state_data,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='state',
            color=value_col,
            color_continuous_scale='Reds',
            title=title,
            labels={value_col: value_col.replace('_', ' ').title()}
        )
        
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=600)
        
        if save_name:
            fig.write_html(f"{self.output_dir}/{save_name}.html")
        
        return fig
    
    def plot_forecast(self, historical_df, forecast_df, value_col, title="Forecast", 
                     group_col='state', save_name=None):
        """
        Plot historical data with forecast
        
        Args:
            historical_df: Historical data
            forecast_df: Forecast data
            value_col: Column name
            title: Plot title
            group_col: Grouping column
            save_name: Filename to save plot
        """
        fig = go.Figure()
        
        # Plot historical data
        for name, group in historical_df.groupby(group_col):
            fig.add_trace(go.Scatter(
                x=group['date'],
                y=group[value_col],
                mode='lines',
                name=f'{name} (Historical)',
                line=dict(width=2)
            ))
        
        # Plot forecast
        forecast_col = f'{value_col}_forecast'
        if forecast_col in forecast_df.columns:
            for name, group in forecast_df.groupby(group_col):
                fig.add_trace(go.Scatter(
                    x=group['date'],
                    y=group[forecast_col],
                    mode='lines',
                    name=f'{name} (Forecast)',
                    line=dict(width=2, dash='dash')
                ))
                
                # Add confidence interval if available
                if f'{value_col}_lower' in group.columns:
                    fig.add_trace(go.Scatter(
                        x=group['date'].tolist() + group['date'].tolist()[::-1],
                        y=group[f'{value_col}_upper'].tolist() + group[f'{value_col}_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{name} Confidence Interval',
                        showlegend=True
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=value_col.replace('_', ' ').title(),
            hovermode='x unified',
            height=500
        )
        
        if save_name:
            fig.write_html(f"{self.output_dir}/{save_name}.html")
        
        return fig
    
    def plot_risk_ranking(self, df, risk_col='aadhaar_fragility_index', top_n=20, save_name=None):
        """
        Plot risk ranking bar chart
        
        Args:
            df: DataFrame with risk scores
            risk_col: Risk column name
            top_n: Number of top regions to show
            save_name: Filename to save plot
        """
        # Get top risky regions
        top_risk = df.nlargest(top_n, risk_col)[['state', 'district', risk_col]].copy()
        top_risk['region'] = top_risk['state'] + ' - ' + top_risk['district']
        
        # Create bar chart
        fig = px.bar(
            top_risk.sort_values(risk_col),
            x=risk_col,
            y='region',
            orientation='h',
            title=f'Top {top_n} High-Risk Regions',
            labels={risk_col: 'Fragility Index', 'region': 'Region'},
            color=risk_col,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(height=600, showlegend=False)
        
        if save_name:
            fig.write_html(f"{self.output_dir}/{save_name}.html")
        
        return fig
    
    def plot_cluster_map(self, df, cluster_col='cluster_kmeans', save_name=None):
        """
        Plot cluster distribution on map
        
        Args:
            df: DataFrame with cluster labels
            cluster_col: Cluster column name
            save_name: Filename to save plot
        """
        # Aggregate by state and cluster
        cluster_data = df.groupby(['state', cluster_col]).size().reset_index(name='count')
        cluster_data[cluster_col] = cluster_data[cluster_col].astype(str)
        
        # Create choropleth
        fig = px.choropleth(
            cluster_data,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='state',
            color=cluster_col,
            title='Regional Clusters',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=600)
        
        if save_name:
            fig.write_html(f"{self.output_dir}/{save_name}.html")
        
        return fig
    
    def plot_sankey_diagram(self, df, save_name=None):
        """
        Create Sankey diagram for Enrolment → Updates flow
        
        Args:
            df: DataFrame with enrolment and update data
            save_name: Filename to save plot
        """
        # Prepare data for Sankey
        # This is a simplified version - customize based on actual data structure
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Enrolments", "Demographic Updates", "Biometric Updates", "No Updates"],
                color=["blue", "green", "orange", "gray"]
            ),
            link=dict(
                source=[0, 0, 0],
                target=[1, 2, 3],
                value=[30, 40, 30]  # Replace with actual values
            )
        )])
        
        fig.update_layout(title_text="Aadhaar Enrolment to Update Flow", font_size=12, height=500)
        
        if save_name:
            fig.write_html(f"{self.output_dir}/{save_name}.html")
        
        return fig
    
    def plot_bubble_chart(self, df, x_col, y_col, size_col, color_col, title="Bubble Chart", save_name=None):
        """
        Create bubble plot for multi-dimensional analysis
        
        Args:
            df: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            size_col: Bubble size column
            color_col: Color column
            title: Plot title
            save_name: Filename to save plot
        """
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col,
            hover_data=['state', 'district'],
            title=title,
            labels={
                x_col: x_col.replace('_', ' ').title(),
                y_col: y_col.replace('_', ' ').title(),
                size_col: size_col.replace('_', ' ').title(),
                color_col: color_col.replace('_', ' ').title()
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=600)
        
        if save_name:
            fig.write_html(f"{self.output_dir}/{save_name}.html")
        
        return fig
    
    def plot_distribution(self, df, value_col, title="Distribution", save_name=None):
        """
        Plot distribution with histogram and KDE
        
        Args:
            df: DataFrame
            value_col: Column to plot
            title: Plot title
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df[value_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'{title} - Histogram', fontweight='bold')
        axes[0].set_xlabel(value_col.replace('_', ' ').title())
        axes[0].set_ylabel('Frequency')
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[value_col].dropna(), vert=True)
        axes[1].set_title(f'{title} - Box Plot', fontweight='bold')
        axes[1].set_ylabel(value_col.replace('_', ' ').title())
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard_summary(self, df):
        """
        Create summary statistics dashboard
        
        Args:
            df: DataFrame with all metrics
        
        Returns:
            Plotly figure with multiple subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Enrolments Over Time', 'Fragility Index Distribution',
                          'State-wise Risk Levels', 'Update Trends'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Time series
        ts_data = df.groupby('date')['total_enrolments'].sum().reset_index()
        fig.add_trace(
            go.Scatter(x=ts_data['date'], y=ts_data['total_enrolments'], 
                      mode='lines', name='Enrolments'),
            row=1, col=1
        )
        
        # Histogram
        if 'aadhaar_fragility_index' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['aadhaar_fragility_index'], name='AFI'),
                row=1, col=2
            )
        
        # Bar chart
        state_risk = df.groupby('state')['aadhaar_fragility_index'].mean().nlargest(10).reset_index()
        fig.add_trace(
            go.Bar(x=state_risk['state'], y=state_risk['aadhaar_fragility_index'], name='Risk'),
            row=2, col=1
        )
        
        # Update trends
        if 'total_updates' in df.columns:
            update_ts = df.groupby('date')['total_updates'].sum().reset_index()
            fig.add_trace(
                go.Scatter(x=update_ts['date'], y=update_ts['total_updates'], 
                          mode='lines', name='Updates'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Aadhaar Observatory Dashboard")
        
        return fig


if __name__ == "__main__":
    print("Visualization Module Ready")
    print("\nAvailable Visualizations:")
    print("1. Time Series Plots (with anomaly markers)")
    print("2. Heatmaps")
    print("3. Choropleth Maps")
    print("4. Forecast Plots")
    print("5. Risk Ranking Charts")
    print("6. Cluster Maps")
    print("7. Sankey Diagrams")
    print("8. Bubble Charts")
    print("9. Distribution Plots")
    print("10. Dashboard Summary")
