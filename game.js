
var width = 700;
var height = 470;
var margin = 50;
var radius = 30;
var svg;
var col_num = 5;
var row_num = 5;
var colWidth = Math.round(width / col_num);
var rowHeight = Math.round(height / row_num);

draw_board = function () {

    svg = d3.selectAll("#game")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    svg.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height)
        .attr("class", "board")
        .attr("rx", 20)
        .attr("ry", 20)
        .style("fill", "#101330");

    for (var rowIndex = 0; rowIndex < row_num; rowIndex++) {
        var yPos = -5 + margin + (rowIndex * rowHeight);

        for (var colIndex = 0; colIndex < col_num; colIndex++) {
            var xPos = 20 + margin + (colIndex *colWidth);

            svg.append("circle")
                .attr("cx", xPos)
                .attr("cy", yPos)
                .attr("r", radius)
                .attr("class", "circle_border")
                .style("fill", "white");

            svg.append("circle")
                .attr("cx", xPos)
                .attr("cy", yPos)
                .attr("r", radius)
                .attr("column", "column" + colIndex)
                .attr("row", "row" + rowIndex)
                .attr("class", "token_spot")
                .style("fill", "white");

        }
    }
}

run_game = function () {

    counter = 0;

    var next_open_spot = [4, 4, 4, 4, 4];

    svg.selectAll(".token_spot")
        .on("mouseover", function() {

            var current_column = d3.select(this).attr("column").slice(-1);
            var row_to_fill = next_open_spot[current_column];

            d3.select("[column=column" + current_column + "][row=row" + row_to_fill + "]")
                .style("fill", "#fdae61");

        })
        .on("mouseout", function() {

            var current_column = d3.select(this).attr("column").slice(-1);
            var row_to_fill = next_open_spot[current_column];

            d3.selectAll("[column=column" + current_column + "]:not(.filled)")
                .style("fill", "white")
                .interrupt()
                .transition();

        })
        .on("click", function() {

            counter = 1-counter;

            var current_column = d3.select(this).attr("column").slice(-1);
            var row_to_fill = next_open_spot[current_column]--;

            d3.select("[column=column" + current_column + "][row=row" + row_to_fill + "]")
                .attr("cy", -30)
                .attr("class", "filled")
                .attr("r", radius-3)
                .style("fill", function() {
                    if (counter == 0) {
                        return "#d53e4f"
                    }
                    else {
                        return "#80cdc1"
                    }
                })
              .transition()
                .duration(500)
                .attr("cy", -5 + margin + (row_to_fill * rowHeight));

            d3.select("[column=column" + current_column + "][row=row" + (row_to_fill-1) + "]")
              .transition()
                .delay(500)
                .duration(0)
                .style("fill", "#fdae61");

        });

}


draw_board ();
run_game();





