package com.sample.www.dto;

import java.util.List;
import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
// js에서 넘겨 받을 값 (자유롭게 커스텀)
public class RequestDto {
	// Common Param
	private int page; // use pageable
	private int size; // use pageeble
	private String sortColumnNaem; // use pageeble
	// ex. Pageable pageable = PageRequest.of(page - 1, size, Sort.by(sortColumnNaem).descending());

	// user custom Param
	private List<Map<String, Object>> list; // sample;
}
