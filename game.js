
var width = 800;
var height = 580;
var margin = 50;
var radius = 30;
var svg;
var col_num = 7;
var row_num = 7;
var colWidth = Math.round(width / col_num);
var rowHeight = Math.round(height / row_num);
var agent = "Minimax"

jQuery.fn.extend({
    disable: function(state) {
        return this.each(function() {
            this.disabled = state;
        });
    }
});

$('#minimax').click(function() {
    $(this).addClass("btn-success");
    $(".btn-default").prop("disabled", true);
    agent = "Minimax";
});

$('#ql').click(function() {
    $(this).addClass("btn-success");
    $(".btn-default").prop("disabled", true);
    agent = "QL";
});

$('#mcts').click(function() {
    $(this).addClass("btn-success");
    $(".btn-default").prop("disabled", true);
    agent = "MCTS";
});

$('#reset').click(function() {
    $(".btn-default").removeClass("btn-success").prop("disabled", false);
    svg.remove();
    draw_board();
    run_game();
})

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
        var yPos = -10 + margin + (rowIndex * rowHeight);

        for (var colIndex = 0; colIndex < col_num; colIndex++) {
            var xPos = 15 + margin + (colIndex *colWidth);

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

    var next_open_spot = [6, 6, 6, 6, 6, 6, 6];
    var board = [[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]];


    svg.selectAll(".token_spot")
        .on("mouseover", function() {
            var current_column = d3.select(this).attr("column").slice(-1);
            var row_to_fill = next_open_spot[current_column];
            

            d3.select("[column=column" + current_column + "][row=row" + row_to_fill + "]")
                .style("fill", function() {
                    if (counter == 0) {
                        return "#9bd8c5"
                    }
                    else {
                        return "#eca7ae"
                    }
                });


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
            make_move = function(current_column, row_to_fill) {
                counter = 1-counter;
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
                    .attr("cy", -10 + margin + (row_to_fill * rowHeight));

            };

            
            var current_column = d3.select(this).attr("column").slice(-1);
            var row_to_fill = next_open_spot[current_column]--;
            
            board[current_column][6 - row_to_fill] = 1;

            make_move(current_column, row_to_fill);
            
            $.ajax({
                    type: "get",
                    url: "?stuff="+JSON.stringify(board)+"&col="+current_column+"&agent="+agent,
                    data: {}
            }).done(function( o ) {
                if (o == -100) {
                    swal({
                        title:"Yay!",
                        text: "You won!",
                        confirmButtonText: "Woot!" 
                    });
                } else {
                    if (o < 0 && o != -100) {
                        var next_column = -1*o - 1;
                        var next_row_to_fill = next_open_spot[next_column]--;
                        board[next_column][6 - next_row_to_fill] = -1;
                        make_move(next_column, next_row_to_fill);
                        setTimeout(function(){ 
                            swal({
                            title:"awwwww...",
                            text: "Computer won!",
                            confirmButtonText: "Maybe next time..." 
                    });; }, 1000);
                    } else {
                        var next_column = o;
                        var next_row_to_fill = next_open_spot[next_column]--;
                        board[next_column][6 - next_row_to_fill] = -1;
                        make_move(next_column, next_row_to_fill);
                    }
                }
            });


        });

}


draw_board ();
run_game();






