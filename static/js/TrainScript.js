$(document).ready(function() {
	//let timerId = setInterval(() => location.reload(), 2000);

	$('#client_info').on('click', function(e) {
		e.preventDefault();
		window.location.href='http://localhost:8891/dashboard';
	});

	$('#train_info').on('click', function(e) {
		e.preventDefault();
		window.location.href='http://localhost:8891/traininfo';
	});
	
	$('#device_info').on('click', function(e) {
		e.preventDefault();
		window.location.href='http://localhost:8891/deviceinfo';
	});
	
	$('#network_info').on('click', function(e) {
		e.preventDefault();
		window.location.href='http://localhost:8891/networkinfo';
	});

	$('#model_info').on('click', function(e) {
		e.preventDefault();
		window.location.href='http://localhost:8891/modelinfo';
	});

	$(".hover").mouseleave(
		function() {
			$(this).removeClass("hover");
		}
	);

	draw_acc_chart(); draw_f1_chart();
	draw_trainingtime_chart(); draw_clientclass_chart();
});
