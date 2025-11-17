import math
import plotly
import plotly.express as px
import polars as pl


def insert_annotation_into_subplot(
        fig: plotly.graph_objs._figure.Figure, 
        ann_df: pl.DataFrame, 
        start_pos: int, 
        end_pos: int, 
        row: int =1, 
        col: int =1
    ):
    """
    Plot genomic annotation in a subplot. 

    """

    df = (
        ann_df
        .with_columns(
            start = pl.col("start").cast(float),
            end = pl.col("end").cast(float)
        )
        .sort("start")
    )

    start_pos = float(start_pos)
    end_pos = float(end_pos)

    axis_num = (row - 1) * (1) + col
    xaxis_name = 'x' if axis_num == 1 else f'x{axis_num}'
    # find numeric x-range for the target subplot axis
    min_x = 0.0
    max_x = float(end_pos - start_pos - 1)
    # min_x = None; max_x = None
    axis_key = 'xaxis' if axis_num == 1 else f'xaxis{axis_num}'
    try:
        xr = fig.layout[axis_key].range
        if xr and len(xr) == 2:
            min_x = float(xr[0]); max_x = float(xr[1])
    except Exception:
        min_x = None
    if min_x is None:
        for tr in fig.data:
            try:
                tr_xaxis = getattr(tr, 'xaxis', None)
                if tr_xaxis is None and axis_num == 1 or tr_xaxis == xaxis_name:
                    xs = getattr(tr, 'x', None)
                    if xs and len(xs) >= 2:
                        try:
                            xs_num = [float(x) for x in xs]
                            min_x = min(xs_num); max_x = max(xs_num)
                            break
                        except Exception:
                            continue
            except Exception:
                continue
    if min_x is None:
        min_x = 0.0; max_x = 1.0
    if end_pos == start_pos:
        def map_x(p): 
            return float(min_x)
    else:
        def map_x(p):
            return float(min_x + (float(p) - start_pos) / (end_pos - start_pos) * (max_x - min_x))
        
    # get paper-domain for target subplot y axis
    yaxis_key = 'yaxis' if axis_num == 1 else f'yaxis{axis_num}'
    try:
        ydom = list(fig.layout[yaxis_key].domain)
    except Exception:
        ydom = [0.0, 1.0]
    y0_dom, y1_dom = float(ydom[0]), float(ydom[1])
    band_height = min(0.36, max(0.12, (y1_dom - y0_dom) * 0.6))
    band_top = y1_dom - 0.02
    band_bottom = band_top - band_height
    # cluster overlaps and pick longest for main bar
    clusters = []
    cur = []
    cur_end = -999
    for r in df.iter_rows(named=True):
        s, e = r['start'], r['end']
        if not cur:
            cur = [r]
            cur_end = e
        else:
            if s <= cur_end:
                cur.append(r); cur_end = max(cur_end, e)
            else:
                clusters.append(cur); cur = [r]; cur_end = e
    if cur: clusters.append(cur)
    main_intervals = []; others = []
    for cluster in clusters:
        if len(cluster) == 1:
            main_intervals.append(cluster[0])
        else:
            best = max(cluster, key=lambda r: (r['end'] - r['start']))
            main_intervals.append(best)
            for r in cluster:
                if r is not best:
                    others.append(r)
    main_intervals = sorted(main_intervals, key=lambda r: r['start'])
    others = sorted(others, key=lambda r: r['start'])
    # colors
    try:
        palette = px.colors.qualitative.Plotly
    except Exception:
        palette = ["#636efa","#ef553b","#00cc96","#ab63fa","#19d3f3","#e763fa","#fecb52"]
    types = list(dict.fromkeys(df['type'].cast(str)))
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(types)}
    # prepare shapes and annotations lists
    shapes = list(fig.layout.shapes) if getattr(fig.layout, 'shapes', None) else []
    annots = list(fig.layout.annotations) if getattr(fig.layout, 'annotations', None) else []
    axis_range = max(1.0, end_pos - start_pos)
    char_data = axis_range * 0.006
    # main bar
    for r in main_intervals:
        x0 = float(max(r['start'], start_pos))
        x1 = float(min(r['end'], end_pos))
        if x1 <= x0: 
            continue
        m0 = map_x(x0)
        m1 = map_x(x1)
        color = color_map.get(str(r['type']), palette[0])
        # col: str = palette[0]
        shapes.append(dict(type='rect', xref=xaxis_name, x0=m0, x1=m1, yref='paper', y0=band_bottom, y1=band_bottom + band_height*0.45, fillcolor=color, line=dict(width=1), layer='above'))
        label = str(r.get('label',''))
        box_width = x1 - x0
        required = max(len(label) * char_data, axis_range * 0.01)
        if box_width >= required:
            annots.append(dict(x=(m0 + m1) / 2, y=band_bottom + band_height*0.225, xref=xaxis_name, yref='paper', text=label, showarrow=False, xanchor='center', yanchor='middle', font=dict(size=12)))
        else:
            annots.append(dict(x=(m0 + m1) / 2, y=band_bottom + band_height*0.45, xref=xaxis_name, yref='paper', ax=0, ay=-10, text=label, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color, xanchor='center', yanchor='bottom', font=dict(size=11)))
    # place others in tracks above the main bar inside the same y-domain
    tracks = []
    track_records = []
    x0: float = 0.0
    x1: float = 0.0
    for r in others:
        x0: float = float(max(r['start'], start_pos))
        x1: float = float(min(r['end'], end_pos))
        if x1 <= x0: 
            continue
        m0 = map_x(x0)
        m1 = map_x(x1)
        placed = False
        for level, track in enumerate(tracks, start=1):
            if all(m1 <= t0 or m0 >= t1 for (t0, t1) in track):
                track.append((m0, m1)); track_records.append((m0, m1, level, (m0 + m1)/2, str(r.get('label','')), color_map.get(str(r['type']), palette[0]))); placed = True; break
        if not placed:
            tracks.append([(m0, m1)]); level = len(tracks)
            track_records.append((m0, m1, level, (m0 + m1)/2, str(r.get('label','')), color_map.get(str(r['type']), palette[0])))
    ntracks = max(1, len(tracks))
    track_slot_height = (band_height * 0.45) / max(1, ntracks)
    for m0, m1, level, center, label, color in track_records:
        y0 = band_bottom + band_height*0.55 + (level - 1) * track_slot_height
        y1 = y0 + track_slot_height * 0.9
        if y1 > (y1_dom - 0.02): 
            y1 = y1_dom - 0.02; y0 = y1 - track_slot_height*0.9
        shapes.append(dict(type='rect', xref=xaxis_name, x0=m0, x1=m1, yref='paper', y0=y0, y1=y1, fillcolor=color, line=dict(width=1), layer='above'))
        # label
        box_data_width = (m1 - m0) * (end_pos - start_pos) / max(1e-9, (max_x - min_x)) if (max_x - min_x)!=0 else 0
        required = max(len(label) * char_data, axis_range * 0.01)

        if ( (m1 - m0) >= 1e-9 and ((x1 - x0) >= required) ):
            annots.append(dict(x=center, y=(y0 + y1)/2, xref=xaxis_name, yref='paper', text=label, showarrow=False, xanchor='center', yanchor='middle', font=dict(size=11)))
        else:
            annots.append(dict(x=center, y=y1, xref=xaxis_name, yref='paper', ax=0, ay=-6, text=label, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color, xanchor='center', yanchor='bottom', font=dict(size=11)))
    fig.layout.shapes = tuple(list(getattr(fig.layout, 'shapes', []) or []) + shapes)
    fig.layout.annotations = tuple(list(getattr(fig.layout, 'annotations', []) or []) + annots)
    # increase figure height slightly to ensure readability
    try:
        orig_h = int(fig.layout.height) if getattr(fig.layout, 'height', None) else 600
    except Exception:
        orig_h = 600
    fig.layout.height = int(orig_h * 1.12)
    return fig