def line_tension_range(cellmap, lower_line_tension, higher_line_tension):
    for edge in range(len(cellmap.edge_df)):
        newValue = random.randrange(lower_line_tension, higher_line_tension)/100
        cellmap.edge_df['line_tension'][edge] = newValue         
    return cellmap