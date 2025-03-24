package com.sample.www.controller.api;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;

@RestController
@RequestMapping("/api/chatBot/")
//@Tag(name = "chatbot API")
public class SendRasaController {

	@PostMapping("chat")
	@Operation(summary = "챗봇 채팅")
	@ApiResponse(responseCode = "200", content = @Content(schema = @Schema(example = "")), description = "status = true/false <br>  "
			+ "code = 1 : 성공/-1 : 실패" + "bot1 :  아이돌말투 / bot2 : 유지원 교사말투 / bot3 : 도도, 차가운 여자 말투" + "message = 반환 받은 텍스트<br>")
	public Map<String, Object> sendRASA(@RequestBody Map<String, String> map) {

		Map<String, Object> result = new HashMap<>();
		String date[];
		String message = "";
		String url = "http://localhost:5005/webhooks/rest/webhook";

		String unicodeString = "";

		try {

			URL obj = new URL(url);
			HttpURLConnection connection = (HttpURLConnection) obj.openConnection();
			connection.setRequestMethod("POST");
			connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");

			// The request body
			String body = "{\"message\": \"" + map.get("message") + "\"}";

//			System.out.println(connection.getURL());

			connection.setDoOutput(true);
			OutputStreamWriter writer = new OutputStreamWriter(connection.getOutputStream());
			writer.write(body);
			writer.flush();
			writer.close();

			BufferedReader br = new BufferedReader(new InputStreamReader(connection.getInputStream()));
			String line;

			StringBuffer response = new StringBuffer();

			while ((line = br.readLine()) != null) {
				response.append(line);
			}
			br.close();

			String re[] = response.toString().split("\"text\":\"");
			for (int i = 1; i < re.length; i++) {
				date = re[i].split("\"}");
//	         date[0] = date[0].replaceAll("\\\\", "");
				// calls the method to extract the message.
//
//			System.out.println(response);
//			System.out.println(date[0]);

				unicodeString = unicodeString.equals("") ? convertString(date[0]) : unicodeString + "\n" + convertString(date[0]);
				unicodeString = unicodeString.replaceAll("\\\\/", "/");
			}
			if (unicodeString.equals("")) {
				result.put("status", false);
				result.put("message", "에러가 발생했습니다.");
				return result;
			}
			System.out.println(response.toString());
			System.out.println(unicodeString);

		} catch (ProtocolException e) {
			throw new RuntimeException(e);
		} catch (MalformedURLException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		try {
			result.put("status", true);
			result.put("text", unicodeString);
		} catch (Exception e) {
			System.out.println(e);
			result.put("status", false);
			result.put("message", "에러가 발생했습니다.");
			return result;
		}
		return result;
	}

	// 유니코드에서 String으로 변환
	public static String convertString(String val) {
		// 변환할 문자를 저장할 버퍼 선언
		StringBuffer sb = new StringBuffer();

		// 글자를 하나하나 탐색한다.
		for (int i = 0; i < val.length(); i++) {

			// 조합이 \\u로 시작하면 6글자를 변환한다. \\uxxxx
			if ('\\' == val.charAt(i) && 'u' == val.charAt(i + 1)) {
				// 그 뒤 네글자는 유니코드의 16진수 코드이다. int형으로 바꾸어서 다시 char 타입으로 강제 변환한다.
				Character r = (char) Integer.parseInt(val.substring(i + 2, i + 6), 16);

				// 변환된 글자를 버퍼에 넣는다.
				sb.append(r);
				// for의 증가 값 1과 5를 합해 6글자를 점프
				i += 5;
			} else {
				// ascii코드면 그대로 버퍼에 넣는다.
				sb.append(val.charAt(i));

			}
		}
		// 결과 리턴
		return sb.toString();
	}

}
