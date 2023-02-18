from dash import html
from engine.widgets import Widget


class Layout:
    """This class defines the layout of the dashboard."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def index(self):
        """This function returns the layout of the dashboard in vanilla html from dash.
        Overwrite this function to create your own layout."""
        # All widgets defined in settings must be added to the vanilla html by calling
        # self.engine.widgets[<widget_id>].get_html_component() at the location you want
        # the widget.
        # Note that dash will give errors if not all widgets are added to the html.
        # For extra components like images -> dash documentation
        settings = self.engine.settings
        return html.Div([
            # Title and logos
            html.Div([
                html.Div([
                    html.Div([
                        # Uncomment this to add a primary logo
                        # html.Img(src=self.engine.app.get_asset_url(
                        #     settings.LAYOUT_SETTINGS['primary-logo']), id='primary-logo')
                    ]),
                    html.Div([
                        html.H1([settings.LAYOUT_SETTINGS['main-title']]),
                    ], id='main-title'),
                ], style={'display': 'flex'}),
                html.Div([
                    html.Div([
                        # Uncomment this to add a secondary logo
                        # html.Img(src=self.engine.app.get_asset_url(
                        #     settings.LAYOUT_SETTINGS['secondary-logo']), id='secondary-logo')
                    ]),
                    html.Div([
                        # Uncomment this to add a tertiary logo
                        # html.Img(src=self.engine.app.get_asset_url(
                        #     settings.LAYOUT_SETTINGS['tertiary-logo']), id='tertiary-logo')
                    ]),
                ], id='logos'),
            ], id='header'),

            # selection options
            html.Div([

                html.Div(
                    Widget.widgets["degree-selector"].get_html_components(),
                    style={'width': '20%', 'display': 'inline-block'}
                ),

                html.Div(
                    Widget.widgets["slice-label-ranking-selector"].get_html_components(),
                    id="rank-dropdown",
                    style={'width': '30%', 'display': 'inline-block'}
                ),

                html.Div(
                    Widget.widgets["slice-label-selector"].get_html_components(),
                    id='slice-label',
                    style={'width': '40%', 'display': 'inline-block'}
                ),
            ], style={'border-width': '2px'}, id='slice-selection-ranking'
            ),

            html.Div(
                html.Div([
                    html.Div([
                        # metric selector + legend
                        html.Div([
                            html.Div(
                                Widget.widgets["metric-selector"].get_html_components(),
                                id='metric-contents'),
                            html.Div(
                                Widget.widgets["legend"].get_html_components(),
                                id='legend-contents')
                        ], id='metric-options'),
                        # tabs + graph
                        html.Div(
                            [
                                html.Div([
                                    html.Div(
                                        Widget.widgets["tabs-bar"].get_html_components()),
                                    html.Div(
                                        Widget.widgets["export-df-button"].get_html_components(), style={"text-align": "right"}),
                                    html.Div(Widget.widgets["tabs-content"].get_html_components(),
                                             style={"overflowX": "scroll"}),
                                ], id='graph-contents'),
                            ],
                            id='graph-area'),
                    ], id='metrics-and-graph'),

                ]), id='output-data-upload'),

            # Region to select models
            html.Div(
                Widget.widgets["model-selector"].get_html_components(), id='uploadzone'),

        ], style={'width': '80%'}, id='body')
