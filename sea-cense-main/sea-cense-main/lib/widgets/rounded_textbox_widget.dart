import '../style.dart';
import 'package:flutter/material.dart';

// ignore: must_be_immutable
class RoundedTextboxWidget extends StatefulWidget {
  TextEditingController controller;
  TextInputType keyboardType;
  bool enabled;
  int minLine;
  int maxLine;
  String labelText;
  double verticalMargin;
  String? Function(dynamic)? validator;
  bool isPassword;

  RoundedTextboxWidget({
    Key? key,
    required this.controller,
    this.keyboardType = TextInputType.text,
    this.enabled = true,
    this.minLine = 1,
    this.maxLine = 1,
    this.verticalMargin = 10,
    required this.labelText,
    this.validator,
    this.isPassword = false,
  }) : super(key: key);

  static InputBorder enabledBorder = const OutlineInputBorder(
    borderSide: BorderSide(
      color: Color(0xffDFDFDF),
    ),
  );

  static InputBorder errorBorder = const OutlineInputBorder(borderSide: BorderSide(color: Colors.red));

  @override
  State<RoundedTextboxWidget> createState() => _RoundedTextboxWidgetState();
}

class _RoundedTextboxWidgetState extends State<RoundedTextboxWidget> {
  bool isVisible = false;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 50,
      width: MediaQuery.of(context).size.width,
      margin: EdgeInsets.symmetric(horizontal: 45, vertical: widget.verticalMargin),
      padding: const EdgeInsets.symmetric(horizontal: 0, vertical: 0.0),
      decoration: BoxDecoration(
        color: Colors.black12,
        borderRadius: BorderRadius.circular(60),
      ),
      child: TextFormField(
        style: const TextStyle(fontSize: 16.0),
        maxLines: widget.maxLine,
        minLines: widget.minLine,
        keyboardType: widget.keyboardType,
        autofocus: false,
        validator: widget.validator,
        controller: widget.controller,
        obscureText: isVisible,
        cursorColor: Colors.purple,
        cursorHeight: 22,
        decoration: InputDecoration(
            border: InputBorder.none,
            hintText: widget.labelText,
            enabled: widget.enabled,
            fillColor: Colors.black12,
            hintStyle: hintStyle,
            contentPadding: const EdgeInsets.all(20),
            suffixIcon: widget.isPassword ? getIcon() : null),
      ),
    );
  }

  getIcon() {
    return GestureDetector(
      onTap: () {
        setState(() {
          isVisible = !isVisible;
        });
      },
      child: isVisible ? const Icon(Icons.visibility) : const Icon(Icons.visibility_off),
    );
  }
}
