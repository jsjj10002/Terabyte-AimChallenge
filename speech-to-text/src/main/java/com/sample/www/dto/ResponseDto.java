package com.sample.www.dto;

import java.util.List;
import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
// js로 넘겨줄 값 (자유롭게 커스텀)
public class ResponseDto {
	// Common Param
	private int resultCode;

	// Pagenation Param
	private int page;
	private int endPage;
	private int totalPage;

	// user custom Param
	private List<Map<String, Object>> list; // sample;
}
