import '../style.dart';
import 'package:flutter/material.dart';

// ignore: must_be_immutable
class RoundedTextboxWidget extends StatelessWidget {
  TextEditingController controller;
  TextInputType keyboardType;
  bool enabled;
  int minLine;
  int maxLine;
  String labelText;
  bool obscureText;
  double verticalMargin;
  String? Function(dynamic)? validator;

  RoundedTextboxWidget({
    Key? key,
    required this.controller,
    this.keyboardType = TextInputType.text,
    this.enabled = true,
    this.minLine = 1,
    this.maxLine = 1,
    this.verticalMargin = 20,
    required this.labelText,
    this.obscureText = false,
    this.validator,
  }) : super(key: key);

  static InputBorder enabledBorder = const OutlineInputBorder(
    borderSide: BorderSide(
      color: Color(0xffDFDFDF),
    ),
  );

  static InputBorder errorBorder = const OutlineInputBorder(borderSide: BorderSide(color: Colors.red));

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 60,
      width: MediaQuery.of(context).size.width,
      margin: EdgeInsets.symmetric(horizontal: 45, vertical: verticalMargin),
      padding: const EdgeInsets.symmetric(horizontal: 0, vertical: 0.0),
      decoration: BoxDecoration(
        color: Colors.black12,
        borderRadius: BorderRadius.circular(60),
      ),
      child: TextFormField(
        style: const TextStyle(fontSize: 16.0),
        maxLines: maxLine,
        minLines: minLine,
        keyboardType: keyboardType,
        autofocus: false,
        validator: validator,
        controller: controller,
        obscureText: obscureText,
        cursorColor: Colors.purple,
        cursorHeight: 15,
        decoration: InputDecoration(
          border: InputBorder.none,
          hintText: labelText,
          enabled: enabled,
          fillColor: Colors.black12,
          hintStyle: hintStyle,
          contentPadding: const EdgeInsets.all(20),
        ),
      ),
    );
  }
}
