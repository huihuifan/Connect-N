
var width = 800;
var height = 500;
var margin = 20;
var radius = 30;

draw_board = function (col_num, row_num) {

    var svg = d3.selectAll("#game")
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
        var yPos = margin + (rowIndex * rowHeight);

        for (var colIndex = 0; colIndex < col_num; colIndex++) {
            var xPos = margin + (colIndex *colWidth);

            svg.append("circle")
                .attr("cx", xPos)
                .attr("cy", yPos)
                .attr("r", radius)
                .style("fill", "white");

        }
    }
}

draw_board (5, 5);


//     drawGrid: function (columns, rows) {

//         var colWidth = Math.round(width / columns);
//         var rowHeight = height / rows;
//         for (var rowIndex = 0; rowIndex < rows; rowIndex++) {
//             grid[rowIndex] = []
//             var yPosition = margin + (rowIndex * rowHeight);
//             for (var colIndex = 0; colIndex <= columns; colIndex++) {
//                 var xPosition = margin + (colIndex * colWidth);
//                 game.drawContainerCircle(xPosition, yPosition);
//                 grid[rowIndex].push({
//                     "x": xPosition,
//                         "y": yPosition,
//                         "full": false
//                 });
//             }
//         }
//         co

// var svg;
// var width, height;
// var margin = 40;
// var game = {};

// var lastRow = 0;

// var grid = {};

// function reset() {
//     game.draw('#animate', 500, 500);
// }

// game = {

//     draw: function (placement, w, h) {

//         d3.select(placement).html("");
//         width = w;
//         height = h;
//         svg = d3.select(placement).append("svg").attr("width", 500).attr("height", 500).append("g");
//         svg.append("rect").attr("x", 0).attr("y", 0).attr("width", width).attr("height", height).style("fill", "#2980b9");

//         game.drawGrid(7, 6);

//         var tokens = 7 * 6;
//         var red = true;
//         for (var tokenCount = 0; tokenCount < tokens; tokenCount++) {

//             var item = game.getNextPosition();
//             var circle = svg.append("circle").attr("class", "row-" + lastRow + " token-" + tokenCount).attr("r", 25).style("fill", function () {
//                 return red ? "#e74c3c" : "#f1c40f"
//             }).attr("cx", 0).attr("cy", -100);

//             // find position to put token
//             d3.select("circle.token-" + tokenCount).transition().duration(100).delay((tokenCount + 1) * 500).attr("cx", item.x);
//             d3.select("circle.token-" + tokenCount).transition().duration(1000).delay((tokenCount + 2) * 500).ease("elastic").attr("cy", item.y);
//             red = !red;
//         }

//         this.emptyGrid();
//     },

//     emptyGrid: function () {
//         for (var rowIndex = 5; rowIndex >= 0; rowIndex--) {
//             d3.selectAll("circle.row-" + rowIndex).transition().duration(1000).delay(25000 - (rowIndex * 50)).ease("elastic").attr("cy", 600);
//         }
//     },

//     drawGrid: function (columns, rows) {

//         var colWidth = Math.round(width / columns);
//         var rowHeight = height / rows;
//         for (var rowIndex = 0; rowIndex < rows; rowIndex++) {
//             grid[rowIndex] = []
//             var yPosition = margin + (rowIndex * rowHeight);
//             for (var colIndex = 0; colIndex <= columns; colIndex++) {
//                 var xPosition = margin + (colIndex * colWidth);
//                 game.drawContainerCircle(xPosition, yPosition);
//                 grid[rowIndex].push({
//                     "x": xPosition,
//                         "y": yPosition,
//                         "full": false
//                 });
//             }
//         }
//         console.log(grid);
//     },

//     drawContainerCircle: function (x, y) {
//         svg.append("circle")
//             .attr("cx", x)
//             .attr("cy", y)
//             .attr("r", 30)
//             .style("fill", "#fff");
//     },

//     getNextPosition: function () {
//         var circleCol = game.getRandomInt(0, 7);

//         for (var rowIndex = 5; rowIndex >= 0; rowIndex--) {
//             if (!grid[rowIndex][circleCol].full) {
//                 // check if this is the next empty row.
//                 grid[rowIndex][circleCol].full = true;
//                 lastRow = rowIndex;
//                 return grid[rowIndex][circleCol];
//             }
//         }
//         // keep going until we find a position...
//         return this.getNextPosition();
//     },

//     getRandomInt: function (min, max) {
//         return Math.floor(Math.random() * (max - min + 1)) + min;
//     }
// }

// game.draw("#animate", 500, 500);


