
var width = 700;
var height = 470;
var margin = 50;
var radius = 30;
var svg;

draw_board = function (col_num, row_num) {

    svg = d3.selectAll("#game")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    svg.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height)
        .style("fill", "#101330");

    var colWidth = Math.round(width / col_num);
    var rowHeight = Math.round(height / row_num);

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

    var next_open_spot = [4, 4, 4, 4, 4];

    svg.selectAll(".token_spot")
        .on("click", function() {

            var current_column = d3.select(this).attr("column").slice(-1);

            var row_to_fill = next_open_spot[current_column]--;

            d3.select("[column=column" + current_column + "][row=row" + row_to_fill + "]")
                .attr("class", "filled")
                .attr("r", radius-3)
                .style("fill", "#d53e4f");

        });

}


draw_board (5, 5);
run_game();






