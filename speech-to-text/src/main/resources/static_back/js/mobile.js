$(document).ready(function() {
	var mobile = isMobile()
	if (mobile == "android") {
		//doCheckToken();
	} else if (mobile == "iphone") {
		//doCheckIOSToken();
	}
})
// ==========================================================================
function isMobile(){
	const UserAgent = navigator.userAgent;
	if (UserAgent.match(/Android|Windows CE|BlackBerry|Symbian|Windows Phone|webOS|Opera Mini|Opera Mobi|POLARIS|IEMobile|lgtelecom|nokia|SonyEricsson/i) != null || UserAgent.match(/LG|SAMSUNG|Samsung/) != null)	{
		return "android";
	} else if (UserAgent.match(/iPhone|iPod/i) != null) {
		return "iphone";
	} else{
		return null;
	}
}
// ==========================================================================
// 안드로이드 함수 호출 예시
function doMessage() {
	if (window.GoodFriendsApp && window.GoodFriendsApp.callMessage) {
		// 해당 브라우저에 MyApp이라는 브릿지가 있고 그 브릿지에 callMessage라는 함수가 있는 경우 호출
		window.GoodFriendsApp.callMessage();
	} else {
		alert("해당 기능은 어플에서만 이용 가능합니다.");
	}
}
// ==========================================================================
// 안드로이드 firebase token 요청하는 함수 예시
function doCheckToken() {
	if (window.TravelmatApp && window.TravelmatApp.CheckToken) {
		window.TravelmatApp.CheckToken();
	} else {
		//alert("해당 기능은 어플에서만 이용 가능합니다.");
	}
}
// 안드로이드 firebase token 받아 java로 보내는 api 함수 예시
function checkToken(usertoken) {
	const data = {usertoken : usertoken};
	const aoo = ajax_object_options("POST", "토큰 받을 api mapping 주소", data);
	ajax(aoo, cts_callback_function, cte_callback_function)
}
// 안드로이드, IOS firebase token 받는 함수의 성공시 콜백 함수
function cts_callback_function(resp) {
	console.log(resp)
}
// 안드로이드, IOS firebase token 받는 함수의 실패시 콜백 함수
function cte_callback_function(resp) {
	console.log(resp)
}
// ==========================================================================
// IOS firebase token 요청하는 함수 예시
function doCheckIOSToken() {
	try{
		webkit.messageHandlers.callbackHandler.postMessage("MessageBody");
	} catch(err) {
		
	}
}
// IOS firebase token 받아 java로 보내는 api 함수 예시
function getFcmTokenToiOS(usertoken) {
	const data = {usertoken : usertoken};
	const aoo = ajax_object_options("POST", "토큰 받을 api mapping 주소", data);
	ajax(aoo, cts_callback_function, cte_callback_function)
}