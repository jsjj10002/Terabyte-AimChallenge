var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
	var SpeechGrammarList = SpeechGrammarList || webkitSpeechGrammarList;
	var SpeechRecognitionEvent = SpeechRecognitionEvent || webkitSpeechRecognitionEvent;
	var diagnosticPara = document.querySelector('.output');
	const tts = new SpeechSynthesisUtterance();
	
	function populateVoiceList() {
		  if (typeof speechSynthesis === 'undefined') {
		    return;
		  }

		  const voices = speechSynthesis.getVoices();

		  const voiceList = voices
		    .filter((voice) => {
		      return voice.lang.includes('en');
		    })
		    .map((voice) => {
		      return `${voice.name} (${voice.lang})`;
		    });
		  console.log(voices);
		}

		populateVoiceList();
		if (
		  typeof speechSynthesis !== 'undefined' &&
		  speechSynthesis.onvoiceschanged !== undefined
		) {
		  speechSynthesis.onvoiceschanged = populateVoiceList;
		}
	
	function sendSpeech() {
	  var recognition = new SpeechRecognition();
	  var speechRecognitionList = new SpeechGrammarList();
	  recognition.grammars = speechRecognitionList;
	  recognition.lang = 'ko-KR';
	  recognition.interimResults = false; // true: 중간 결과를 반환, false: 최종 결과만 반환
	  recognition.continious = false; // true: 음성인식을 계속해서 수행, false: 음성인식을 한번만 수행
	  recognition.maxAlternatives = 1;
	
	  recognition.start();
	
	  recognition.onresult = function(event) {
	    var speechResult = event.results[0][0].transcript.toLowerCase();
	    console.log('Confidence: ' + event.results[0][0].confidence);
	    console.log('Speech Result: ' + speechResult);
	    //diagnosticPara.textContent = 'Speech received: ' + speechResult + '.';
	    diagnosticPara = $('.textBox');
	    userText = $('.userText');
	    userText.html(speechResult);
	    var aoo = ajax_object_options('POST', '/api/chatBot/chat', { message : speechResult});
	    ajax(aoo, function (resp) {
	    	var text = resp.text;
	    	diagnosticPara.html(text);
	    	if (text.includes("주문하신") || text.includes("수정")) {
		    	$('.orderListBox').empty();
		    	var data = {
					content : '',
					coffeeKind : 0,
					coffeeCount : 0,
					price : 0
				}
				var menu = ["에스프레소", "아메리카노", "카푸치노", "카페라떼", "바닐라라떼", "카라멜마끼야또", "카페모카", "아포카토",
							"토마토주스", "키위주스", "망고스무디", "딸기스무디", "쿠키앤크림",
							"레몬에이드", "복숭아아이스티", "허브티", "말차라떼", "초콜릿라떼"];
				for (var i = 0; i < menu.length; i++) {
					data = searchCoffee(text, menu[i], data);
				}
	    		if (text.includes('결제')) {
	    			setTimeout(() => {
	                    $('.contents').css('display', 'none');
	                    $('.payment').css('display', 'flex');
	        
	                    setTimeout(() => {
	                        $('.payment').css('display', 'none');
	                        $('.paymentEnd').css('display', 'flex');
	                        setTimeout(() => {
		                        $('.paymentEnd').css('display', 'none');
		                        $('.contents').css('display', 'flex');
		                        $('.orderListBox').empty();
		        	    		$('#coffeeCount').text(0);
		        	    		$('#coffeePrice').text(0);
		        	    		userText.html('');
		                    }, 15000);
	                        window.speechSynthesis.cancel();
	                        diagnosticPara.html('안녕하세요. 서비스 지원을 위한 제니입니다.<br>원하시는 서비스를 말씀해 주세요.');
	                        tts.lang = 'ko-KR';
	            	    	tts.pitch = 1;
	            	    	tts.rate = 1;
	            	    	tts.text = '주문이 성공적으로 진행되었습니다. 카드를 회수해 주세요. 저희 매장을 이용해 주셔서 감사합니다. 주문번호를 확인 하신 후 안내된 번호에 맞춰 카운터로 오시면 됩니다.';
	            	    	tts.volume = 1;
	            	    	window.speechSynthesis.speak(tts);
	                    }, 7000);
	                }, 1000);
	    		}

	    		if (text.includes('취소')) {
	    			$('.orderListBox').empty();
    	    		$('#coffeeCount').text(0);
    	    		$('#coffeePrice').text(0);
		        	userText.html('');
	    		}
	    		$('.orderListBox').html(data.content);
	    		$('#coffeeCount').text(data.coffeeCount);
	    		$('#coffeePrice').text(data.price.toLocaleString('ko-KR'));
	    	}
	    	tts.lang = 'ko-KR';
	    	tts.pitch = 1;
	    	tts.rate = 1;
	    	tts.text = resp.text;
	    	tts.volume = 1;
	    	window.speechSynthesis.speak(tts);
	    }, function (resp) {
	    	alert('에러 발생');
	    	console.log(resp);
	    })
	  }
	
	  recognition.onaudiostart = function(event) {
	      //Fired when the user agent has started to capture audio.
	      console.log('SpeechRecognition.onaudiostart');
	  }
	
	  recognition.onaudioend = function(event) {
	      //Fired when the user agent has finished capturing audio.
	      console.log('SpeechRecognition.onaudioend');
	  }
	
	  recognition.onend = function(event) {
	      //Fired when the speech recognition service has disconnected.
	      console.log('SpeechRecognition.onend');
	  }
	
	  recognition.onnomatch = function(event) {
	      //Fired when the speech recognition service returns a final result with no significant recognition. This may involve some degree of recognition, which doesn't meet or exceed the confidence threshold.
	      console.log('SpeechRecognition.onnomatch');
	  }
	
	  recognition.onsoundstart = function(event) {
	      //Fired when any sound — recognisable speech or not — has been detected.
	      console.log('SpeechRecognition.onsoundstart');
	  }
	
	  recognition.onsoundend = function(event) {
	      //Fired when any sound — recognisable speech or not — has stopped being detected.
	      console.log('SpeechRecognition.onsoundend');
	      const micSrc = $('.mic').children('img')
	      micSrc.attr('src','/img/off.png');
          $('.mic').removeClass('on');
	  }
	
	  recognition.onspeechstart = function (event) {
	      //Fired when sound that is recognised by the speech recognition service as speech has been detected.
	      console.log('SpeechRecognition.onspeechstart');
	  }
	  recognition.onstart = function(event) {
	      //Fired when the speech recognition service has begun listening to incoming audio with intent to recognize grammars associated with the current SpeechRecognition.
	      console.log('SpeechRecognition.onstart');
	  }
	}

	function searchCoffee(text, coffee, data) {
		if (text.includes(coffee)) {
	    			var coffeeCount = 0;
	    			var textSplit = text.split(coffee);
	    			data.coffeeKind++;
	    			var coffeePrice = coffeeGetPrice(coffee);
	    			if (textSplit.length > 1) {
	    				var zanSplit = textSplit[1].split("잔")[0];
	    				switch(zanSplit.trim()) {
    					case "한":
    						coffeeCount = coffeeCount + 1;
    						break;
    					case "두":
    						coffeeCount = coffeeCount + 2;
    						break;
    					case "세":
    						coffeeCount = coffeeCount + 3;
    						break;
    					case "네":
    						coffeeCount = coffeeCount + 4;
    						break;
    					case "다섯":
    						coffeeCount = coffeeCount + 5;
    						break;
    					case "여섯":
    						coffeeCount = coffeeCount + 6;
    						break;
    					case "일곱":
    						coffeeCount = coffeeCount + 7;
    						break;
    					case "여덟":
    						coffeeCount = coffeeCount + 8;
    						break;
    					case "아홉":
    						coffeeCount = coffeeCount + 9;
    						break;
    					case "열":
    						coffeeCount = coffeeCount + 10;
    						break;
    					default:
    						break;
	    				}
	    				data.coffeeCount = data.coffeeCount + coffeeCount
	    				data.price = data.price + coffeeCount * coffeePrice;
	    				data.content = data.content + '<div class="orderList">'
	                    	+'<div class="number">'+data.coffeeKind+'</div>'
	                    	+'<div class="menu">'+coffee+'</div>'
	                    	+'<div class="menu">'+coffeeCount+'</div>'
	                    	+'<div class="menu">'+(coffeeCount*coffeePrice).toLocaleString('ko-KR')+' <span>원</span></div>'
	                    	+'</div>'
	    			}
	    		} 
	    		return data;
	}
	
	function coffeeGetPrice(coffee) {
		var price = 0;
		switch(coffee.trim()) {
    					case "에스프레소":
    						price = 3800
    						break;
    					case "아메리카노":
    						price = 4300
    						break;
    					case "카푸치노":
    						price = 4500
    						break;
    					case "카페라떼":
    						price = 4500
    						break;
    					case "바닐라라떼":
    						price = 4800
    						break;
    					case "카라멜마끼야또":
    						price = 4800
    						break;
    					case "카페모카":
    						price = 4800
    						break;
    					case "아포카토":
    						price = 5000
    						break;
    					case "토마토주스":
    						price = 5500
    						break;
    					case "키위주스":
    						price = 5500
    						break;
    					case "망고스무디":
    						price = 5800
    						break;
    					case "딸기스무디":
    						price = 5800
    						break;
    					case "쿠키앤크림":
    						price = 5800
    						break;
    					case "레몬에이드":
    						price = 5500
    						break;
    					case "복숭아아이스티":
    						price = 5000
    						break;
    					case "허브티":
    						price = 5000
    						break;
    					case "말차라떼":
    						price = 5800
    						break;
    					case "초콜릿라떼":
    						price = 5800
    						break;
    					default:
    						break;
		}
		return price;
	}