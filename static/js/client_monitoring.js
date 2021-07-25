function requestData() {
	$.ajax({
		url: '/dashboard',
		method: "GET",
		success: function(point) {
			console.log(point);
			//setTimeout(requestData, 60000);
		},
		cache: false
	});
}

$(document).ready(function() {
	$('#restart').on('click', function(e) {
		e.preventDefault();
		console.log("click restart");
		requestData();
	});
});
