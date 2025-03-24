$(document).ready(function(){
	$('body').keyup();
	setInterval(async () => AccessClipboardData(), 300); // 0.3초마다 클립보드 초기화
})

document.oncontextmenu = function() {return false;} // 우클릭 방지
document.ondragstart = function() {return false;}; // 마우스 드래그 금지
document.onselectstart = function() {return false;}; // 선택 금지, 마우스 드래그 금지
// document.onkeydown = function(){return false;}; // 키 입력 금지 (왠만하면 사용 금지)

// 스크린샷 키 검사
document.addEventListener("keyup", function (e) {
    //var keyCode = e.keyCode ? e.keyCode : e.which; // e.keyCode == 44 (print screen keyCode)
	if (e.key == 'PrintScreen') {
		const agent = window.navigator.userAgent.toLowerCase();
		try {
			if (agent.indexOf("chrome") > -1) {
				navigator.clipboard.writeText('');		
			} else {			
				window.clipboardData.setData('text', "Access Restricted");
			}
		} catch (err) {
			//console.log(err)
		}
		console.log('Screenshots disabled!')
        stopPrntScr();
	}
});

// 스크린샷 방지 함수 (완벽히는 못 막음)
function stopPrntScr() {
	var inpFld = document.createElement("input");
	inpFld.setAttribute("value", ".");
	inpFld.setAttribute("width", "0");
	inpFld.style.height = "0px";
	inpFld.style.width = "0px";
	inpFld.style.border = "0px";
	document.body.appendChild(inpFld);
	inpFld.select();
	document.execCommand("copy")
	inpFld.remove(inpFld)
}

// 클립보드 초기화 함수
async function AccessClipboardData() {
	try {
		const agent = window.navigator.userAgent.toLowerCase();
		if (agent.indexOf("chrome") > -1) {
			await navigator.clipboard.writeText('');		
		} else {			
			window.clipboardData.setData('text', "Access Restricted");
		}
	} catch (err) {
		//console.log(err)
	}
}

// 마우스가 벗어났을때 (필요할 때 사용)
/*document.addEventListener('mouseleave', function() {
	$('body').css('display', 'none');
})*/

// 마우스가 올라왔을때 (필요할 때 사용)
/*document.addEventListener('mouseenter', function() {
	$('body').css('display', 'block');
})*/

// 크롬 탭 전환시
document.addEventListener('visibilitychange', function() {
	if ($('body').css('display') == 'block') {		
		$('body').css('display', 'none');
	} else {
		$('body').css('display', 'block');
	}
})

// 클릭한 html 요소 확인
function what_click(event) {
	console.log(event.target)
}