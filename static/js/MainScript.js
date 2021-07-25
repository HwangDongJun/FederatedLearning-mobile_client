$(document).ready(function() {
	//let timerId = setInterval(() => location.reload(), 2000);

	$('#restart').on('click', function(e) {
		e.preventDefault();
		location.reload();
	});

	$('#delete').on('click', function(e) {
		e.preventDefault();
		let delete_result = delete_client($('#delete').attr("name"));
		location.reload();
	});

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
});

function delete_client(client_name) {
	$.ajax({
		url: '/delete',
		method: "GET",
		data: {cn: client_name},
		success: function(point) {
			console.log(point);
		},
		cache: false
	});

	return "success";
}
