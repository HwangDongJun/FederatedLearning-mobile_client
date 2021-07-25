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

	draw_heapsize_chart();
	draw_temperature_chart(); draw_cpu_freq_chart();
});
